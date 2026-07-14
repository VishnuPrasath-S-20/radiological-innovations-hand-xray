import sys
sys.path.insert(0, '/home/xo/code/radiological-innovations-hand-xray')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import os

from src.data.datasets import HandXRayCancerDataset
from src.data.transforms import get_cancer_transforms
from src.models.multi_task_model import CancerModel
from resource_monitor import AutoResourceManager

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # --- Initialize Auto-Resource Manager ---
    monitor = AutoResourceManager(
        gpu_temp_limit=82,
        cpu_temp_limit=85,
        vram_limit_percent=0.92,
        ram_limit_percent=90,
        verbose=True
    )
    print("🛡️  AutoResourceManager ACTIVE (GPU + CPU Monitoring)")
    
    # --- Data ---
    train_transform, valid_transform = get_cancer_transforms(224)
    train_ds = HandXRayCancerDataset(split='train', transform=train_transform)
    valid_ds = HandXRayCancerDataset(split='valid', transform=valid_transform)
    
    # --- Initial Dataloader ---
    current_batch_size = 32
    current_num_workers = 1
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=current_batch_size, 
        shuffle=True, 
        num_workers=current_num_workers, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"📦 Initial Batch Size: {current_batch_size}")
    print(f"🔧 Initial Num Workers: {current_num_workers}")
    
    # --- Model ---
    model = CancerModel(backbone='efficientnet_b0', pretrained=True).to(device)
    
    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # ============================================
    # --- CHECKPOINT: RESUME FROM BEST MODEL ---
    # ============================================
    start_epoch = 0
    best_val_acc = 0.0

    # --- Option 1: Full checkpoint resume (most reliable) ---
    if os.path.exists('checkpoint_cancer.pth.tar'):
        print("🔄 Found full checkpoint. Loading to resume exactly...")
        checkpoint = torch.load('checkpoint_cancer.pth.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch} (loss: {checkpoint['loss']:.4f})")
        # Load best accuracy from file if it exists
        if os.path.exists('best_val_acc.txt'):
            with open('best_val_acc.txt', 'r') as f:
                best_val_acc = float(f.read().strip())
            print(f"✅ Previous best accuracy: {best_val_acc:.4f}")
        else:
            best_val_acc = 0.0

    # --- Option 2: Best model weights only (fallback) ---
    elif os.path.exists('best_cancer_model.pth'):
        print("🔄 Found existing best cancer model. Loading weights to resume training...")
        model.load_state_dict(torch.load('best_cancer_model.pth'))
        start_epoch = 0  # Set to your last completed epoch + 1 (e.g., 5 if you did 5 epochs)
        # Load previous best accuracy
        if os.path.exists('best_val_acc.txt'):
            with open('best_val_acc.txt', 'r') as f:
                best_val_acc = float(f.read().strip())
            print(f"✅ Previous best accuracy: {best_val_acc:.4f}")
        else:
            best_val_acc = 0.0
            print("⚠️ No best accuracy file found. Starting fresh.")
        print(f"✅ Weights loaded. Resuming from Epoch {start_epoch + 1}.")
    else:
        print("🚀 No existing model found. Starting training from scratch.")

    print(f"\n🔥 Starting Cancer Detection Training (Auto-Resource ON)...\n")
    
    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, 50):
        # --- Check resource at the start of every epoch ---
        new_batch, new_workers = monitor.manage_resources(current_batch_size, current_num_workers)
        if new_batch != current_batch_size or new_workers != current_num_workers:
            current_batch_size = new_batch
            current_num_workers = new_workers
            train_loader = DataLoader(
                train_ds, 
                batch_size=current_batch_size, 
                shuffle=True, 
                num_workers=current_num_workers, 
                pin_memory=True
            )
            print(f"🔄 DataLoader recreated: batch_size={current_batch_size}, num_workers={current_num_workers}")
        
        # --- Training ---
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        epoch_restarted = False
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/50")):
            # Check resources every 10 batches
            if batch_idx % 10 == 0:
                new_batch, new_workers = monitor.manage_resources(current_batch_size, current_num_workers)
                if new_batch != current_batch_size or new_workers != current_num_workers:
                    current_batch_size = new_batch
                    current_num_workers = new_workers
                    train_loader = DataLoader(
                        train_ds, 
                        batch_size=current_batch_size, 
                        shuffle=True, 
                        num_workers=current_num_workers, 
                        pin_memory=True
                    )
                    print(f"🔄 Resource change mid-epoch! Restarting...")
                    epoch_restarted = True
                    break
            
            # Forward / Backward
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        if epoch_restarted:
            print(f"🔄 Epoch {epoch+1} restarted due to resource change. Skipping validation.")
            continue
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # --- Metrics ---
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        
        # --- Save best model (only if improvement) ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cancer_model.pth')
            with open('best_val_acc.txt', 'w') as f:
                f.write(str(best_val_acc))
            print(f"✅ Best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            print(f"ℹ️ No improvement (best is {best_val_acc:.4f})")
        
        # --- SAVE CHECKPOINT (full state) ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }
        torch.save(checkpoint, 'checkpoint_cancer.pth.tar')
        print(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        scheduler.step()
        print(f"📊 {monitor.get_display_string()}\n")
    
    print("\n🎉 Cancer Detection Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
