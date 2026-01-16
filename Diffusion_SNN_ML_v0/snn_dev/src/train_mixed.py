import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.baseline import CellDataset, SimpleSNN, train as baseline_train

class SyntheticDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform
        self.class_map = {c: i for i, c in enumerate(config.CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["file"]).convert("RGB")
        label = self.class_map[item["class"]]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def train_mixed(synthetic_manifest="synthetic_manifest.json"):
    print("--- Training Mixed (Real + Synthetic) SNN ---")
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # 1. Real Dataset
    real_ds = CellDataset(config.TRAIN_JSON, transform=transform)
    
    # 2. Synthetic Dataset
    if not Path(synthetic_manifest).exists():
        print(f"Error: {synthetic_manifest} not found. Run generation first.")
        return
        
    syn_ds = SyntheticDataset(synthetic_manifest, transform=transform)
    
    # 3. Combine
    mixed_ds = ConcatDataset([real_ds, syn_ds])
    print(f"Combined Dataset Size: {len(mixed_ds)} (Real: {len(real_ds)}, Synthetic: {len(syn_ds)})")
    
    # Use the same training logic as baseline but with the larger dataset
    # We might need to adjust epochs or LR, but let's keep it simple for comparison
    # We'll pass the mixed dataset to a modified version of the baseline train function
    from src import baseline
    
    # Patch baseline globals for the session
    baseline.BATCH_SIZE = 32 # Larger batch for more data
    
    # Re-run training (this is a bit hacky but keeps code DRY)
    # Ideally baseline.py should have a more flexible train function
    # Let's just implement a local version
    
    train_dl = DataLoader(mixed_ds, batch_size=32, shuffle=True)
    val_ds   = CellDataset(config.VAL_JSON,   transform=transform)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    
    from src.baseline import SimpleSNN, DEVICE, CLASS_WEIGHTS, LR, EPOCHS, TIME_STEPS
    from sklearn.metrics import f1_score, classification_report
    import torch.nn as nn
    
    model = SimpleSNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            spk_rec, mem_rec = model(imgs)
            loss_val = torch.zeros((1), dtype=torch.float, device=DEVICE)
            for step in range(TIME_STEPS):
                loss_val += criterion(mem_rec[step], labels)
            
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss_val.item()
            
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                spk_rec, _ = model(imgs)
                spike_count = spk_rec.sum(0)
                _, preds = spike_count.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "mixed_snn.pth")
            print("\nClassification Report (Improved):")
            print(classification_report(all_targets, all_preds, target_names=config.CLASSES, zero_division=0))

if __name__ == "__main__":
    train_mixed()
