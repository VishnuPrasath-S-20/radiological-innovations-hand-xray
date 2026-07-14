# Radiological Innovations - Hand X-ray Multi-Task Learning

A Multi-Task Deep Learning framework for automated **bone age estimation** and **cancer/abnormality detection** in hand X-ray radiographs using PyTorch.

---

## 📊 Results

| Model | Metric | Best Value |
| :--- | :--- | :--- |
| **Bone Age (Age + Sex)** | Validation Loss | **181.95** |
| **Bone Age** | Val MAE | **181.74 months** (~15.1 years) |
| **Cancer Detection** | Test Accuracy | **97.59%** |
| **Cancer Detection** | Test AUC | **0.976** |

---

## 🧠 Models

### 1. Bone Age Model
- **Architecture**: EfficientNet-B0 (Multi-Task)
- **Tasks**: Age (regression) + Sex (classification)
- **Best Val Loss**: 181.95
- **Training Epochs**: 14

### 2. Cancer Detection Model
- **Architecture**: EfficientNet-B0 (Binary Classification)
- **Task**: Benign vs Malignant
- **Test Accuracy**: 97.59%
- **Test AUC**: 0.976

---

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/VishnuPrasath-S-20/radiological-innovations-hand-xray.git
cd radiological-innovations-hand-xray
