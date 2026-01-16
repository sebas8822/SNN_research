import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src import data_loader
from src import preprocessing

# --- Configuration ---
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
BETA = 0.95  # SNN decay rate
TIME_STEPS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class weights from previous analysis (approx)
# Prophase: 0.34, Metaphase: 0.65, Anaphase: 1.39, Telophase: 1.62
CLASS_WEIGHTS = torch.tensor([0.34, 0.65, 1.39, 1.62], dtype=torch.float32).to(DEVICE)

class CellDataset(Dataset):
    def __init__(self, json_path, loader=None, transform=None):
        self.loader = loader if loader else data_loader.CocoLoader(train_json_path=json_path)
        self.annotations = self.loader.get_annotations()
        self.transform = transform
        self.class_map = {c: i for i, c in enumerate(config.CLASSES)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        cls_name = self.loader.get_category_name(ann["category_id"])
        label = self.class_map.get(cls_name, 0)
        
        # Load and Preprocess on the fly (or load cached patches if valid)
        # For baseline, let's load cached patches to ensure consistency with pipeline
        patch_path = config.PATCH_CACHE_TRAIN / cls_name / f"patch_{ann['id']}.png"
        
        if patch_path.exists():
            img = Image.open(patch_path).convert("RGB")
        else:
            # Fallback to raw extraction (slow) - mainly for robust dev
            # For this baseline script, we assume patches are generated via 'make run-pipeline'
            # If not found, create a blank image (should not happen if pipeline run)
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))

        if self.transform:
            img = self.transform(img)
            
        return img, label

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Simple convnet
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1  = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2  = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
        
        self.flatten = nn.Flatten()
        
        # Calculate flat size: 96 -> 46 -> 23 -> 19 -> 9 (roughly)
        # Let's do a dummy pass to check size, or calculate:
        # 96 -4 = 92 /2 = 46. 46 -4 = 42 /2 = 21. 32*21*21
        self.fc1 = nn.Linear(32 * 21 * 21, 4)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []
        mem3_rec = []

        for step in range(TIME_STEPS):
            cur1 = self.pool1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc1(self.flatten(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

def train():
    print("--- Training Baseline SNN ---")
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # We use the 'train' JSON for training, and split it? 
    # Or strict Train/Val from files. Config says separate files.
    train_ds = CellDataset(config.TRAIN_JSON, transform=transform)
    val_ds   = CellDataset(config.VAL_JSON,   transform=transform)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleSNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # SNN input usually static image repeated over time
            loss_val = 0
            optimizer.zero_grad()
            
            spk_rec, mem_rec = model(imgs)
            
            # Sum spikes over time for classification logits? 
            # Or Mean pooling. Typically snntorch uses Rate coding or last step.
            # Using CrossEntropy on average membrane potential or summed spikes.
            # Let's use summed spikes (rate code).
            loss_val = torch.zeros((1), dtype=torch.float, device=DEVICE)
            for step in range(TIME_STEPS):
                loss_val += criterion(mem_rec[step], labels)
            
            loss_val.backward()
            optimizer.step()
            total_loss += loss_val.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                spk_rec, _ = model(imgs)
                # Count spikes over time
                spike_count = spk_rec.sum(0)
                _, preds = spike_count.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "baseline_snn.pth")
            
            print("\nClassification Report (Best):")
            print(classification_report(all_targets, all_preds, target_names=config.CLASSES, zero_division=0))

if __name__ == "__main__":
    train()
