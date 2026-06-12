import sys
import re

with open('train_fed_finetune.py', 'r') as f:
    content = f.read()

# 1. Replace local_train_one_round
old_local_train = re.search(r'def local_train_one_round\(.*?return avg_loss, train_acc, accumulated_grads\n', content, re.DOTALL).group(0)

new_local_train = """def local_train_one_round(
    encoder,
    classifier,
    loader,
    optimizer,
    criterion,
    args,
    global_params,
    freeze_encoder,
    comm_round=0,
    scaffold_state=None,
    global_centroids=None,
):
    import torch.nn.functional as F
    import torch
    from models.wrappers import PatchEncoderWrapper
    
    encoder.train()
    total_loss = 0.0
    valid_batches = 0
    correct = 0
    total = 0
    
    # 1. Check if we are in Round 0 (Cold Start)
    is_cold_start = (global_centroids is None or len(global_centroids) == 0)
    
    # Accumulators for the new centroids
    class_sums = {i: None for i in range(3)}
    class_counts = {i: 0 for i in range(3)}

    for images, labels in loader:
        images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)
        
        # In Cold Start, we do NOT train the encoder. We only extract features.
        if is_cold_start:
            with torch.no_grad():
                if isinstance(encoder, PatchEncoderWrapper):
                    features = encoder(images)
                else:
                    features = encoder(images, return_patches=False)
                
                if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
                features = F.normalize(features, p=2, dim=1)
                loss = torch.tensor(0.0)
        else:
            # Round 1+: Execute the Cosine Softmax Loss
            optimizer.zero_grad()
            if isinstance(encoder, PatchEncoderWrapper):
                features = encoder(images)
            else:
                features = encoder(images, return_patches=False)
                
            if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
            
            # Build frozen anchors
            anchors = torch.stack([global_centroids[k] for k in range(3)]).to(args.device)
            anchors = F.normalize(anchors, p=2, dim=1)
            
            # Compute logits and loss
            tau = 0.07
            sim_logits = torch.matmul(features, anchors.T)
            loss = F.cross_entropy(sim_logits / tau, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            
            preds = sim_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        total += images.size(0)

        # Accumulate features for centroid calculation
        with torch.no_grad():
            for i in range(len(labels)):
                c = labels[i].item()
                if class_sums[c] is None:
                    class_sums[c] = torch.zeros_like(features[i])
                class_sums[c] += features[i].detach()
                class_counts[c] += 1

    local_centroids = {}
    for c in range(3):
        if class_counts[c] > 0:
            local_centroids[c] = F.normalize(class_sums[c] / class_counts[c], p=2, dim=0)

    avg_loss = total_loss / max(1, valid_batches)
    train_acc = 100.0 * correct / max(total, 1) if not is_cold_start else 0.0
    return avg_loss, train_acc, local_centroids, None
"""
content = content.replace(old_local_train, new_local_train)

# 2. Replace evaluate_global
old_eval_global = re.search(r'@torch\.no_grad\(\)\ndef evaluate_global\(.*?return val_acc, val_loss, auc, diagnostics\n', content, re.DOTALL).group(0)

new_eval_global = """@torch.no_grad()
def evaluate_global(
    encoder,
    classifier,
    loader,
    device,
    num_classes,
    class_weights,
    global_centroids=None,
):
    import torch.nn.functional as F
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support
    from models.wrappers import PatchEncoderWrapper

    encoder.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    if global_centroids is None or len(global_centroids) == 0:
        diagnostics = {
            "unweighted_val_loss": 0.0,
            "balanced_acc": 0.0,
            "per_class_recall": [0.0]*3,
            "per_class_f1": [0.0]*3,
            "per_class_support": [0]*3,
            "prediction_hist": [0]*3,
            "feature_norm_mean": 0.0,
            "feature_std_mean": 0.0,
        }
        return 0.0, 0.0, 0.0, diagnostics
        
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        if isinstance(encoder, PatchEncoderWrapper):
            features = encoder(images)
        else:
            features = encoder(images, return_patches=False)
            
        if features.dim() > 2: features = features.mean(dim=[2,3]) if features.dim()==4 else features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)
        
        logits = torch.zeros((features.size(0), 3), device=device)
        for c, centroid in global_centroids.items():
            anchor = centroid.to(device)
            logits[:, c] = F.cosine_similarity(features, anchor.unsqueeze(0))
            
        probs = torch.softmax(logits / 0.07, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu())
        
    val_acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    
    all_probs_np = torch.cat(all_probs).numpy() if all_probs else np.zeros((0, num_classes))
    all_labels_np = np.array(all_labels)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
        else:
            auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr")
    except Exception:
        auc = 0.0
        
    labels_range = list(range(num_classes))
    try:
        _, recall, f1, support = precision_recall_fscore_support(
            all_labels_np, all_preds_np, labels=labels_range, zero_division=0
        )
        balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        pred_hist = np.bincount(all_preds_np, minlength=num_classes).astype(int).tolist()
    except Exception:
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        support = np.zeros(num_classes)
        balanced_acc = 0.0
        pred_hist = [0]*num_classes
        
    diagnostics = {
        "unweighted_val_loss": 0.0,
        "balanced_acc": float(balanced_acc),
        "per_class_recall": recall.astype(float).tolist() if hasattr(recall, 'astype') else recall,
        "per_class_f1": f1.astype(float).tolist() if hasattr(f1, 'astype') else f1,
        "per_class_support": support.astype(int).tolist() if hasattr(support, 'astype') else support,
        "prediction_hist": pred_hist,
        "feature_norm_mean": 1.0,
        "feature_std_mean": 0.0,
    }
    return val_acc, 0.0, auc, diagnostics
"""
content = content.replace(old_eval_global, new_eval_global)

# 3. Replace main loop sections
content = content.replace("for comm_round in range(start_round, args.max_rounds):", "global_centroids = None\n    for comm_round in range(start_round, args.max_rounds):")

old_local_loop = re.search(r'        client_losses    = \[\]\n        client_train_acc = \[\]\n.*?client_train_acc\.append\(tacc\)\n', content, re.DOTALL).group(0)

new_local_loop = """        client_losses    = []
        client_train_acc = []
        client_local_centroids = {}

        # ---- Local training for each client ----
        for cid in range(args.n_clients):
            enc   = client_encoders[cid]
            cls   = client_classifiers[cid]
            opt   = client_optimizers[cid]

            for pg in opt.param_groups:
                if pg.get("group_name") == "lora_encoder":
                    pg["lr"] = current_lr * pg.get("scale", 1.0)
                elif pg.get("is_encoder", False):
                    pg["lr"] = enc_lr * pg.get("scale", 1.0)
                else:  # "classifier"
                    pg["lr"] = cls_lr

            local_criterion = criterion

            loss, tacc, local_centroids, accumulated_grads = local_train_one_round(
                enc, cls, client_loaders[cid], opt,
                local_criterion, args,
                global_params if use_fedprox else None,
                freeze_encoder,
                comm_round=comm_round,
                scaffold_state=None,
                global_centroids=global_centroids,
            )
            client_losses.append(loss)
            client_train_acc.append(tacc)
            client_local_centroids[cid] = local_centroids
"""
content = content.replace(old_local_loop, new_local_loop)

# 4. Replace server aggregation
old_server_agg = re.search(r'        encoder_update_norms = \[.*?head_diag = classifier_head_diagnostics\(global_classifier\)\n', content, re.DOTALL).group(0)

new_server_agg = """        encoder_update_norms = [
            model_update_norm(global_encoder, client_encoders[i])
            for i in range(args.n_clients)
        ]
        classifier_update_norms = [0.0] * args.n_clients

        # ---- Model aggregation ----
        average_models(global_encoder,    client_encoders,    encoder_agg_weights,
                       server_momentum=server_momentum_enc)
        
        # 2. Aggregate the new Global Centroids (Prototypes)
        import torch.nn.functional as F
        new_global_centroids = {}
        for c in range(args.num_classes):
            total_c_samples = sum(counts.get(c, 0) for counts in client_class_counts)
            if total_c_samples > 0:
                c_sum = None
                for client_idx in range(args.n_clients):
                    if client_idx in client_local_centroids and c in client_local_centroids[client_idx]:
                        weight = client_class_counts[client_idx].get(c, 0) / total_c_samples
                        feat = client_local_centroids[client_idx][c].to(args.device)
                        if c_sum is None:
                            c_sum = torch.zeros_like(feat)
                        c_sum += feat * weight
                if c_sum is not None:
                    new_global_centroids[c] = F.normalize(c_sum, p=2, dim=0)
                
        global_centroids = new_global_centroids

        # ---- Broadcast back ----
        broadcast_global_to_clients(global_encoder,    client_encoders)

        # ---- Global evaluation ----
        val_acc, val_loss, val_auc, eval_diag = evaluate_global(
            global_encoder, global_classifier,
            test_loader, args.device, args.num_classes,
            eval_class_weights,
            global_centroids=global_centroids,
        )
        head_diag = {}
"""
content = content.replace(old_server_agg, new_server_agg)

with open('train_fed_finetune.py', 'w') as f:
    f.write(content)
print("Patch successful!")
