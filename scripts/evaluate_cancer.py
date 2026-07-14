import sys
sys.path.insert(0, '/home/xo/code/radiological-innovations-hand-xray')

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.data.datasets import HandXRayCancerDataset
from src.data.transforms import get_cancer_transforms
from src.models.multi_task_model import CancerModel

def evaluate_cancer(model_path='best_cancer_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # --- Load test data ---
    _, test_transform = get_cancer_transforms(224)
    test_ds = HandXRayCancerDataset(split='test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # --- Load model ---
    model = CancerModel(backbone='efficientnet_b0', pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Predict ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'])

    print("\n📊 Test Set Evaluation")
    print("=" * 50)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_cancer()
