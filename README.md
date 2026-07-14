# 🩻 Radiological Innovations - Hand X-ray Multi-Task Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Multi-Task Deep Learning** framework for automated **bone age estimation** and **cancer/abnormality detection** in hand X-ray radiographs using PyTorch and EfficientNet-B0.

---

## 📊 Results

| Model | Metric | Best Value |
| :--- | :--- | :--- |
| **Bone Age (Age + Sex)** | Validation Loss | **181.95** |
| **Bone Age** | Val MAE | **181.74 months** (~15.1 years) |
| **Cancer Detection** | Test Accuracy | **97.59%** |
| **Cancer Detection** | Test AUC | **0.976** |

---

## 🧠 Model Architecture

### 1. Bone Age Model (`MultiTaskModel`)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Tasks**:
  - **Age**: Regression (MSE Loss)
  - **Sex**: Binary Classification (CrossEntropy Loss)
- **Training Epochs**: 14 (early stopping at plateau)
- **Best Validation Loss**: 181.95

### 2. Cancer Detection Model (`CancerModel`)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Task**: Binary Classification (Benign vs Malignant)
- **Training Epochs**: 6 (converged early)
- **Test Accuracy**: 97.59%
- **Test AUC**: 0.976
- **Confusion Matrix**:
```text
  [[476  12]   → Benign: 476 correct, 12 misclassified
   [  9 375]]  → Malignant: 375 correct, 9 misclassified
 ``` 
---

## 📁 Project Structure

```text
radiological-innovations-hand-xray/
├── src/
│   ├── data/
│   │   ├── datasets.py          # PyTorch Dataset classes
│   │   └── transforms.py        # Albumentations augmentations
│   └── models/
│       └── multi_task_model.py  # MultiTaskModel & CancerModel
├── scripts/
│   ├── train_age.py             # Age model training
│   ├── train_cancer.py          # Cancer model training
│   ├── inference.py             # Inference on single images
│   └── evaluate_cancer.py       # Test set evaluation
├── resource_monitor.py          # AutoResourceManager (GPU/CPU monitoring)
├── check_csv.py                 # Data validation script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files (venv, models, checkpoints)
└── README.md                    # This file
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/VishnuPrasath-S-20/radiological-innovations-hand-xray.git](https://github.com/VishnuPrasath-S-20/radiological-innovations-hand-xray.git)
cd radiological-innovations-hand-xray
```

### 2. Create a Virtual Environment

```bash
# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA (GPU Support)

```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# For CUDA 12.1
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Datasets

| Dataset | Source | Notes |
| --- | --- | --- |
| **Bone Age (RSNA)** | [Kaggle RSNA Bone Age](https://www.kaggle.com/competitions/rsna-bone-age) | ~14,000 hand X-rays |
| **Cancer Detection** | [Kaggle Bone Cancer Detection](https://www.kaggle.com/datasets/ziya07/bone-cancer-detection-dataset) | ~8,800 X-rays (all bones) |

**Folder Structure After Download:**

```text
/run/media/.../Dataset/radiological-innovations-hand-xray/
├── age/
│   ├── train/          # 12,611 images + atrain.csv
│   └── valid/          # 1,425 images + avalid.csv
└── cancer/
    ├── train/          # 7,057 images + ctrain.csv
    ├── valid/          # 882 images + cvalid.csv
    └── test/           # 872 images + ctest.csv
```

> ⚠️ **Note:** Remember to update the `BASE_PATH` variable inside `src/data/datasets.py` to match your local dataset partition path before running the code.

---

## 🚀 Training

### Train Bone Age Model

```bash
python scripts/train_age.py
```

### Train Cancer Detection Model

```bash
python scripts/train_cancer.py
```

Both scripts include:

* ✅ **AutoResourceManager** (GPU/CPU temperature monitoring)
* ✅ **Checkpoint Resume** (resume from best model or full checkpoint)
* ✅ **Learning Rate Scheduling** (CosineAnnealingLR)
* ✅ **Validation** after every epoch

---

## 🔬 Inference

### Run Inference on a Single Image

```bash
python scripts/inference.py
```

### Python API Example

```python
from scripts.inference import predict_age_and_sex, predict_cancer

# Bone Age + Sex Prediction
age, sex = predict_age_and_sex('hand_xray.png')
print(f"Age: {age:.1f} months ({age/12:.1f} years)")
print(f"Sex: {sex}")

# Cancer Detection Prediction
cancer, prob_mal, prob_ben = predict_cancer('bone_xray.png')
print(f"Cancer Status: {cancer}")
print(f"Malignant Probability: {prob_mal*100:.1f}%")
print(f"Benign Probability: {prob_ben*100:.1f}%")
```

---

## 🛡️ AutoResourceManager (System Monitoring)

The project includes an internal real-time system monitor to manage hardware resources dynamically:

| Condition | Action |
| --- | --- |
| **GPU Temp > 82°C** | Reduces batch size or pauses training for 30 seconds |
| **CPU Temp > 85°C** | Automatically reduces `num_workers` or pauses for 15 seconds |
| **VRAM Utilization > 92%** | Explicitly flushes cache memory via `torch.cuda.empty_cache()` |
| **RAM Utilization > 90%** | Restricts and reduces system `num_workers` allocations |

This ensures **stable model training loops** when working directly on engineering laptops (benchmarked on an Acer Nitro laptop containing an RTX 4050 GPU configuration).

---

## 📈 Evaluation

### Evaluate Cancer Model on Test Set

```bash
python scripts/evaluate_cancer.py

```

---

## 📦 Requirements

See `requirements.txt` for the full dependency checklist.

```text
opencv-python>=4.7.0
Pillow>=9.5.0
albumentations>=1.3.0
timm>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
```

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vishnu Prasath S** Master of Engineering (M.E.) Student in Computer Science and Engineering

[GitHub](https://github.com/VishnuPrasath-S-20)

---

## 🙏 Acknowledgements

* [RSNA Bone Age Challenge](https://www.kaggle.com/competitions/rsna-bone-age) for the bone age dataset.
* [Kaggle Bone Cancer Detection Dataset](https://www.kaggle.com/datasets/ziya07/bone-cancer-detection-dataset) for the anomaly classification dataset.
* [PyTorch](https://pytorch.org/) for providing the foundational deep learning framework.
* [timm](https://github.com/huggingface/pytorch-image-models) for the clean EfficientNet pretrained model collection backbones.
* [Albumentations](https://albumentations.ai/) for the high-performance pixel-level augmentation strategies.

---

## ⭐ Star the Repository

If you find this framework useful for your research or medical vision projects, please consider dropping a star! ⭐
