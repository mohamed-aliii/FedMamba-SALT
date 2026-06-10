import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_fed_finetune import parse_args, build_models, build_client_dataloaders
from eval.linear_probe import get_eval_transform, get_train_transform
from augmentations.retina_dataset import RetinaDataset
from utils.ckpt_compat import safe_torch_load

def evaluate_calibration(encoder, loader, global_centroids, device, class_weights):
    encoder.eval()
    correct_standard = 0
    correct_calibrated = 0
    total = 0
    
    # 1. Build the frozen prototype matrix
    anchors = torch.stack([global_centroids[k] for k in range(3)]).to(device)
    anchors = F.normalize(anchors, p=2, dim=1)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Extract features
            features = encoder(images)
            if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
            
            # Standard Cosine Similarity
            sim_logits = torch.matmul(features, anchors.T)
            preds_standard = torch.argmax(sim_logits, dim=1)
            
            # Calibrated Cosine Similarity (Applying the margin)
            calibrated_logits = sim_logits * class_weights.to(device)
            preds_calibrated = torch.argmax(calibrated_logits, dim=1)
            
            # Track accuracies
            correct_standard += (preds_standard == labels).sum().item()
            correct_calibrated += (preds_calibrated == labels).sum().item()
            total += labels.size(0)
            
    print(f"--- INFERENCE DIAGNOSTIC RESULTS ---")
    print(f"Standard Accuracy (Uncalibrated): {100. * correct_standard / total:.2f}%")
    print(f"Margin-Calibrated Accuracy:       {100. * correct_calibrated / total:.2f}%")

def reconstruct_centroids(encoder, client_loaders, num_classes, device):
    """
    The checkpoint doesn't save global_centroids.
    We must mathematically reconstruct them from the training set.
    """
    print("=> Reconstructing global centroids from the training set (takes ~1-2 minutes)...")
    encoder.eval()
    class_sums = {i: None for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for cid, loader in enumerate(client_loaders):
            print(f"   Processing Client {cid+1}/{len(client_loaders)}...")
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
                features = F.normalize(features, p=2, dim=1)
                
                for i in range(len(labels)):
                    c = labels[i].item()
                    if class_sums[c] is None:
                        class_sums[c] = torch.zeros_like(features[i])
                    class_sums[c] += features[i]
                    class_counts[c] += 1
    
    global_centroids = {}
    for c in range(num_classes):
        global_centroids[c] = F.normalize(class_sums[c] / class_counts[c], p=2, dim=0)
    print("=> Centroid reconstruction complete!")
    return global_centroids

def train_and_evaluate_linear_probe(encoder, client_loaders, test_loader, num_classes, device):
    print("\n" + "="*55)
    print("=> Extracting all Federated Features to Memory...")
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for cid, loader in enumerate(client_loaders):
            for images, labels in loader:
                images = images.to(device)
                features = encoder(images)
                if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
                
                # We do NOT L2 normalize here because nn.Linear can learn the optimal scale.
                # If we normalize, it restricts the magnitude which can hamper the linear probe.
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"=> Extracted {all_features.size(0)} features of dimension {all_features.size(1)}.")
    
    # Create a unified, shuffled dataloader
    dataset = torch.utils.data.TensorDataset(all_features, all_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    print("=> Training Central Linear Probe over Unified Shuffled Features (10 Epochs)...")
    head = torch.nn.Linear(all_features.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        head.train()
        total_loss = 0
        samples = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            samples += labels.size(0)
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/10 - Probe Loss: {total_loss / samples:.4f}")
        
    print("\n=> Evaluating Central Linear Probe on Test Set...")
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
            logits = head(features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    print(f"--- LINEAR PROBE RESULTS ---")
    print(f"Trained Linear Probe Accuracy: {100. * correct / total:.2f}%")
    print("="*55)

if __name__ == "__main__":
    # Extract --ckpt_path manually so parse_args doesn't crash on it
    ckpt_path = None
    if "--ckpt_path" in sys.argv:
        idx = sys.argv.index("--ckpt_path")
        ckpt_path = sys.argv[idx + 1]
        del sys.argv[idx:idx+2]
    else:
        print("Error: --ckpt_path is required")
        sys.exit(1)

    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    args._label_map = None
    if args.dataset == 'covidfl' and args.num_classes == 2:
        args._label_map = {0: 0, 1: 0, 2: 1}
        
    print("=> Building dataloaders...")
    train_transform = get_train_transform(dataset=args.dataset)
    client_loaders, _ = build_client_dataloaders(args, train_transform)
    
    eval_transform = get_eval_transform(dataset=args.dataset)
    test_ds = RetinaDataset(
        data_path=args.data_path,
        phase="test",
        split_type="central",
        split_csv="test.csv",
        transform=eval_transform,
        label_map=args._label_map,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"=> Loading checkpoint: {ckpt_path}")
    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    
    print("=> Building model...")
    encoder, _ = build_models(args)
    # FIX: The key is 'encoder_state_dict', not 'global_encoder'
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    
    # FIX: The checkpoint doesn't save global_centroids. Reconstruct them.
    global_centroids = reconstruct_centroids(encoder, client_loaders, args.num_classes, device)
    
    # Your explicit class weights from the logs
    class_weights = torch.tensor([0.734, 1.021, 1.518], dtype=torch.float32)
    
    # Run the zero-cost evaluation
    print("\n" + "="*55)
    evaluate_calibration(encoder, test_loader, global_centroids, device, class_weights)
    
    # Run the Central Linear Probe Training (zero-cost federated, minimal central compute)
    train_and_evaluate_linear_probe(encoder, client_loaders, test_loader, args.num_classes, device)