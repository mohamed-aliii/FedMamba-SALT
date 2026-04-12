import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import get_student_transform_val
from eval.linear_probe import load_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
ckpt_path = "/content/fedmamba_salt/outputs/retina_centralized/ckpt_latest.pth"
encoder = load_encoder(ckpt_path, device, freeze=True)

# Load data subset
transform = get_student_transform_val(dataset="retina")
ds = RetinaDataset("/content/drive/MyDrive/Retina", split_csv="central/test.csv", transform=transform)
subset = torch.utils.data.Subset(ds, range(1000))
loader = DataLoader(subset, batch_size=256, shuffle=False)

# Extract features
encoder.eval()
raw_feats = []
labels = []
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        feat = encoder(x)
        raw_feats.append(feat.cpu())
        labels.append(y.cpu())

X_raw = torch.cat(raw_feats, dim=0)
X_norm = F.normalize(X_raw, dim=-1)
Y = torch.cat(labels, dim=0)

print(f"Extracted features: {X_raw.shape}, Labels: {Y.shape}")
print(f"Stats raw: mean={X_raw.mean().item():.4f}, std={X_raw.std().item():.4f}")

# Train loop func
def train_lr(X, Y, epochs=50, lr=1e-3, name="Raw"):
    classifier = nn.Linear(768, 2)
    opt = Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nTraining on {name} features (lr={lr}, epochs={epochs}):")
    for ep in range(epochs):
        opt.zero_grad()
        logits = classifier(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()
        
        acc = (logits.argmax(dim=1) == Y).float().mean().item()
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:2d} | loss={loss.item():.4f} | acc={acc*100:.2f}%")

train_lr(X_raw, Y, epochs=100, lr=1e-3, name="Raw")
train_lr(X_raw, Y, epochs=100, lr=1e-2, name="Raw, High LR")
train_lr(X_norm, Y, epochs=100, lr=1e-3, name="Normalized")
train_lr(X_norm, Y, epochs=100, lr=1e-2, name="Normalized, High LR")
train_lr(X_norm, Y, epochs=100, lr=1e-1, name="Normalized, Huge LR")
