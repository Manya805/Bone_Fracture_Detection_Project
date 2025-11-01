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


# 🦴 Fracture Detection using CNN and XGBoost

## 🧠 Why These Methods

- **CNN** learns **spatial visual cues** such as bone discontinuities, texture patterns, and shape irregularities directly from raw pixel values — ideal for detecting fractures in X-ray imagery.  

- **EfficientNet + XGBoost** provides a **compact, interpretable, and data-efficient pipeline**, combining pretrained visual feature extraction with a robust tree-based classifier.  

- This **hybrid comparison** highlights the **trade-offs** between:
  - End-to-end **deep learning** (CNN: better recall and visual understanding)
  - and **boosted-tree models** (XGBoost: better interpretability, speed, and accuracy on smaller datasets).

---

## ⚙️ How to Run the Project

Follow these steps to train, evaluate, and visualize both models.

---

### 🧩 Install Dependencies
```bash
pip install -r requirements.txt

🧠 Train CNN

To train the Convolutional Neural Network on your dataset:

python fracture_detect.py


This script:

Loads and augments training and validation X-ray images.

Trains a CNN model for binary classification (fractured vs not fractured).

Saves the trained model as fracture_detection_model.h5 and also converts it to a .tflite model for lightweight inference.

🌲 Train XGBoost on Deep Features

Train the XGBoost classifier on deep visual features extracted from EfficientNetB0:

python tree_on_features.py


This script:

Uses EfficientNetB0 (pretrained on ImageNet) to extract 1280-dimensional image embeddings.

Saves embeddings in features/deep_features.npz.

Trains the XGBoost model and saves it as xgb_model_meta.joblib.

📊 Evaluate and Compare

Run the evaluation script to compare CNN and XGBoost results:

python evaluate.py


This script:

Computes metrics including Accuracy, Precision, Recall, F1-Score, AUC, MSE, MAE, RMSE, and Loss.

Generates and saves:

📈 Performance metrics bar chart

🧩 Confusion matrices

🚦 ROC curve comparison

Stores all results inside the results/ directory.

🧾 Notebook Version (Optional)

If you prefer running in an interactive notebook:

evaluate.ipynb


Run each cell to reproduce training, evaluation, and visualization outputs step-by-step.

💻 Launch Streamlit App

Launch the web app to interactively test both models:

streamlit run app.py


The app allows you to:

Upload X-ray images (.jpg, .jpeg, .png)

Get predictions and probabilities from both CNN and XGBoost models

Visually compare their outputs

📂 Output Location

All plots, metrics, and models are automatically saved to:

results/


Includes:

model_comparison.csv

metrics_comparison.png

confusion_matrices.png

roc_comparison.png

✅ Make sure to train both models (fracture_detect.py and tree_on_features.py) before running evaluate.py or launching the Streamlit app.

🧪 Experiments and Results Summary
📉 Quantitative Comparison
Model	Accuracy	Precision	Recall	F1-Score	AUC	MSE	MAE	RMSE	Loss
CNN	0.6083	0.6268	0.8583	0.7245	0.7432	0.2455	0.3649	0.4955	0.7072
XGBoost	0.7500	0.7966	0.7833	0.7899	0.8253	0.1685	0.3188	0.4105	0.1685

Observation:
XGBoost outperformed CNN in most metrics — particularly in precision, AUC, and error-based measures, showing better generalization and lower prediction error.

🎨 Visual Results
📈 Performance Metrics Comparison

🧩 Confusion Matrices

🚦 ROC Curve Comparison

🔍 Interpretation of Plots

CNN Confusion Matrix:
CNN shows a higher false positive rate — it tends to overpredict “fracture” cases, leading to high recall but lower precision.

XGBoost Confusion Matrix:
Achieves a better balance between true positives and true negatives, indicating improved generalization.

ROC Curves:
XGBoost has a higher AUC (0.8253), meaning it distinguishes between classes more effectively.

Bar Chart:
XGBoost consistently performs better in accuracy, F1-score, and AUC, while CNN leads slightly in recall (higher sensitivity).

⚙️ Hyperparameter Experiments
Model	Parameter	Tested Values	Best
CNN	Learning Rate	0.001 – 0.0001	0.0001
CNN	Dropout	0.3 – 0.5	0.5
XGBoost	n_estimators	100 – 300	200
XGBoost	max_depth	4 – 8	6
🧭 Conclusion

XGBoost achieved higher accuracy (0.75) and better AUC (0.825) than the CNN, making it more robust for smaller datasets.

CNN maintained higher recall (0.8583), which is critical for ensuring fewer missed fracture detections — at the cost of more false alarms.

These results show that deep feature + tree-based hybrid approaches can outperform pure CNNs in medical imaging tasks with limited data.

🧩 Future Work

Combine both models using ensemble techniques for improved overall precision-recall trade-off.

Add explainability (Grad-CAM for CNN, SHAP for XGBoost) for clinical interpretability.

Extend testing to multi-class fracture localization and other radiology datasets.

📚 References

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

Kaggle Dataset: Bone Fracture X-ray Classification — https://www.kaggle.com

Chollet, F. (2017). Deep Learning with Python. Manning Publications.

TensorFlow Documentation — https://www.tensorflow.org

XGBoost Documentation — https://xgboost.readthedocs.io
