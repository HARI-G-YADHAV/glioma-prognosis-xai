# 🧠 Empowering Glioma Prognosis with Transparent Machine Learning & Explainable AI

## 📌 Project Overview
Gliomas, especially **Glioblastoma Multiforme (GBM)**, are aggressive brain tumors that require accurate prognosis for effective treatment.  
This project leverages **Machine Learning (ML)** and **Explainable AI (XAI)** techniques to build a **reliable and interpretable glioma prediction system** using a dataset of **889 patients**.

We identified key predictors such as:
- **IDH1 mutation**
- **TP53 mutation**
- **Age at diagnosis**
- **EGFR mutation**

Our best-performing model (**XGBoost**) achieved:
- **88% Accuracy**
- **92% AUC**

---

## 🎯 Motivation
Traditional prognosis methods rely heavily on manual interpretation of medical and genetic data, which can be:
- Inconsistent
- Time-consuming  
Our goal was to **automate prognosis** while ensuring **transparency** so that clinicians can trust and understand predictions.

---

## 📂 Dataset & Preprocessing
**Dataset Size**: 889 patients  
**Features**: 23 clinical and genetic attributes

**Preprocessing Steps:**
- Data balancing
- Standardization for consistent scaling
- Feature selection using:
  - Pearson’s correlation
  - Mutual Information
  - Principal Component Analysis (PCA)

---

## 🤖 Machine Learning & Deep Learning Models
### ML Models:
- Random Forest
- XGBoost ✅ *(Best performer)*
- LightGBM
- CatBoost

### Deep Learning Models:
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

### Ensemble:
- Stacking method combining model outputs for enhanced performance

---

## 🧩 Explainable AI (XAI) Techniques
We used **four XAI tools** to ensure transparency and interpretability:
- **SHAP** – Quantified feature contributions using game theory
- **LIME** – Local explanations approximating black-box models
- **ELI5** – Simplified model predictions for human understanding
- **QLattice** – Explored feature interactions via probabilistic graphs

---

## 📊 Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC (Area Under the Curve)

**Best Result**:  
- **XGBoost** → *88% accuracy*, *92% AUC*

---

## 🛠️ Implementation
- **Language:** Python
- **Libraries:** scikit-learn, TensorFlow, SHAP, LIME, ELI5
- **Training/Test Split:** 80:20
- **Hyperparameter Tuning:** Grid Search

---

## 🚀 How to Run
1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/glioma-prognosis-xai.git
cd glioma-prognosis-xai
