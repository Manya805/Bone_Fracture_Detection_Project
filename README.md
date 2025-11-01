# Bone_Fracture_Detection_Project

# Fracture Detection using CNN and XGBoost

## 1. Title and Short Description
This project implements **automated bone fracture detection** using X-ray images.  
It compares two machine learning approaches:
- A **Convolutional Neural Network (CNN)** trained end-to-end on images.
- An **XGBoost model** trained on deep features extracted via **EfficientNetB0**.

### Why it matters
Fracture detection is a critical diagnostic step in orthopedics. Manual X-ray interpretation can be error-prone and time-consuming, especially in emergency settings.  
This system aims to assist radiologists by automating fracture identification and comparing traditional and deep learning approaches.

**Key Outcome:**  
XGBoost achieved better accuracy (0.75) and AUC (0.8253) compared to CNN (accuracy = 0.6083, AUC = 0.7432), demonstrating that hybrid ML + deep feature methods can outperform standalone CNNs on smaller datasets.

---

## 2. Dataset Source
- **Dataset:** Publicly available X-ray dataset of fractured and non-fractured bones.
- **Total Samples:** ≈ 2,800 images.
- **Classes:**  
  - `fractured`  
  - `not fractured`
- **Data Split:** 70 % train, 20 % validation, 10 % test.
- **Source:** Open-access medical image datasets (e.g., Kaggle “Bone Fracture X-ray” dataset).

### Preprocessing
- Corrupted images removed.
- Normalized to pixel range `[0, 1]`.
- Image augmentation applied: random rotation (±10°), zoom (±5 %), horizontal flips.
- Deep features (1280-dimensional vectors) extracted using EfficientNetB0’s global pooling layer.

---

## 3. Methods

### Architecture Overview
| Stage | Description |
|-------|--------------|
| **CNN Model** | 3× Conv2D + MaxPooling layers → Dense(128) + Dropout(0.5) → Sigmoid output. |
| **Feature Extractor** | EfficientNetB0 pretrained on ImageNet (used to generate embeddings). |
| **XGBoost Classifier** | Trained on extracted embeddings to predict fracture presence. |

### Workflow Diagram
```mermaid
flowchart LR
A[Input X-Ray Image] --> B{Preprocessing}
B --> C[128×128 → CNN]
B --> D[224×224 → EfficientNetB0]
C --> E[CNN Classifier]
D --> F[Feature Vector]
F --> G[XGBoost Classifier]
E --> H[Fracture / Not Fracture]
G --> H
H --> I[Evaluation & Comparison]
