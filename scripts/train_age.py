import sys
sys.path.insert(0, '/home/xo/code/radiological-innovations-hand-xray')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from src.data.datasets import HandXRayAgeDataset
from src.data.transforms import get_age_transforms
from src.models.multi_task_model import MultiTaskModel
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
    train_transform, valid_transform = get_age_transforms(224)
    train_ds = HandXRayAgeDataset(split='train', transform=train_transform)
    valid_ds = HandXRayAgeDataset(split='valid', transform=valid_transform)
    
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
    model = MultiTaskModel(backbone='efficientnet_b0', pretrained=True).to(device)
    
    # --- Loss & Optimizer ---
    age_criterion = nn.MSELoss()
    sex_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # ============================================
    # --- CHECKPOINT: RESUME FROM BEST MODEL ---
    # ============================================
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists('best_age_model.pth'):
        print("🔄 Found existing best model. Loading weights to resume training...")
        model.load_state_dict(torch.load('best_age_model.pth'))
        # Set this to the LAST epoch you completed + 1 (e.g., 10 if you finished Epoch 10)
        start_epoch = 10  # Adjust this number based on your last completed epoch
        # Load the previous best loss so we don't overwrite it
        if os.path.exists('best_val_loss.txt'):
            with open('best_val_loss.txt', 'r') as f:
                best_val_loss = float(f.read().strip())
            print(f"✅ Previous best loss: {best_val_loss:.4f}")
        else:
            best_val_loss = float('inf')
            print("⚠️ No best loss file found. Starting fresh.")
        print(f"✅ Weights loaded. Resuming training from Epoch {start_epoch + 1}.")
    else:
        print("🚀 No existing model found. Starting training from scratch.")
    
    print("\n🔥 Starting Age+Sex Training (Auto-Resource ON)...\n")
    
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
        epoch_restarted = False
        
        for batch_idx, (images, ages, sexes) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/50")):
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
            ages = ages.to(device)
            sexes = sexes.to(device)
            
            optimizer.zero_grad()
            age_pred, sex_pred = model(images)
            loss = age_criterion(age_pred, ages) + 0.3 * sex_criterion(sex_pred, sexes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if epoch_restarted:
            print(f"🔄 Epoch {epoch+1} restarted due to resource change. Skipping validation.")
            continue
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_age_loss = 0
        with torch.no_grad():
            for images, ages, sexes in valid_loader:
                images = images.to(device)
                ages = ages.to(device)
                sexes = sexes.to(device)
                age_pred, sex_pred = model(images)
                loss_age = age_criterion(age_pred, ages)
                loss_sex = sex_criterion(sex_pred, sexes)
                loss = loss_age + 0.3 * loss_sex
                val_loss += loss.item()
                val_age_loss += loss_age.item()
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(valid_loader)
        avg_val_age_loss = val_age_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Age MAE={avg_val_age_loss:.4f}")
        
        # --- Save best model (only if improvement) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_age_model.pth')
            with open('best_val_loss.txt', 'w') as f:
                f.write(str(best_val_loss))
            print(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            print(f"ℹ️ No improvement (best is {best_val_loss:.4f})")
        
        # --- SAVE CHECKPOINT (full state) ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }
        torch.save(checkpoint, 'checkpoint.pth.tar')
        print(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        scheduler.step()
        print(f"📊 {monitor.get_display_string()}\n")
    
    print("\n🎉 Age+Sex Training Complete!")

if __name__ == "__main__":
    main()
