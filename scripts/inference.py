import sys
sys.path.insert(0, '/home/xo/code/radiological-innovations-hand-xray')

import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os

from src.models.multi_task_model import MultiTaskModel, CancerModel

def predict_age_and_sex(image_path, model_path='best_age_model.pth', device='cpu'):
    """Predict bone age and sex from a hand X-ray"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None, None
    
    model = MultiTaskModel(backbone='efficientnet_b0', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Image not found: {image_path}")
        return None, None
    
    image = np.array(image)
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        age, sex_logits = model(img_tensor)
    
    age_months = age.item()
    sex_pred = torch.argmax(sex_logits, dim=1).item()
    sex_label = 'Male' if sex_pred == 1 else 'Female'
    
    return age_months, sex_label

def predict_cancer(image_path, model_path='best_cancer_model.pth', device='cpu'):
    """Predict cancer risk from an X-ray"""
    
    if not os.path.exists(model_path):
        print(f"❌ Cancer model not found: {model_path}")
        return None, None, None
    
    model = CancerModel(backbone='efficientnet_b0', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Image not found: {image_path}")
        return None, None, None
    
    image = np.array(image)
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
    
    prob_malignant = probs[0, 1].item()
    prob_benign = probs[0, 0].item()
    prediction = 'Malignant' if prob_malignant > 0.5 else 'Benign'
    
    return prediction, prob_malignant, prob_benign

if __name__ == "__main__":
    print("🔬 Inference Ready!")
    print("Usage:")
    print("  age, sex = predict_age_and_sex('path/to/image.png')")
    print("  cancer, prob_mal, prob_ben = predict_cancer('path/to/image.png')")
    print()
    
    # Example: if you have a test image, uncomment and modify the path below:
    # test_image = '/path/to/your/test_xray.png'
    # age, sex = predict_age_and_sex(test_image)
    # if age is not None:
    #     print(f"Age: {age:.1f} months ({age/12:.1f} years)")
    #     print(f"Sex: {sex}")
    # 
    # cancer, prob_mal, prob_ben = predict_cancer(test_image)
    # if cancer is not None:
    #     print(f"Cancer Prediction: {cancer}")
    #     print(f"Malignant Probability: {prob_mal*100:.1f}%")
    #     print(f"Benign Probability: {prob_ben*100:.1f}%")
