# 🧠 Alzheimer's Disease Early Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade ML pipeline for early-stage Alzheimer's classification, deployed as a REST API.**

[Features](#-features) · [Pipeline](#-pipeline) · [API Reference](#-api-reference) · [Results](#-results) · [Setup](#-setup) · [Project Structure](#-project-structure)

</div>

---

## Overview

This project implements a supervised machine learning classification system for early detection of Alzheimer's Disease using clinical, cognitive, and neuroimaging features. The full automated pipeline covers data ingestion, preprocessing, exploratory data analysis (EDA), feature engineering, model training with cross-validation and hyperparameter tuning, and deployment as a production Flask REST API.

The goal is to provide clinicians and researchers with a transparent, explainable, and accurate risk-scoring tool that flags patients in the early stages — when intervention is most effective.

---

## ✨ Features

- **End-to-end ML pipeline** — from raw CSV ingestion to live API predictions
- **Multiple classifiers** benchmarked — Random Forest, XGBoost, SVM, Logistic Regression
- **Stratified 5-fold cross-validation** with `GridSearchCV` hyperparameter tuning
- **SHAP-based explainability** — per-prediction feature importance scores
- **Production Flask REST API** with input validation, error handling, and logging
- **Automated EDA reports** — correlation heatmaps, class balance analysis, distribution plots
- **Dockerisable** — ready for containerised deployment

---

## 📊 Results

| Model | Accuracy | F1 Score | AUC-ROC | CV Std |
|---|---|---|---|---|
| **XGBoost** ✅ | **94.8%** | **0.941** | **0.976** | ±0.012 |
| Random Forest | 91.4% | 0.903 | 0.947 | ±0.018 |
| SVM (RBF) | 88.2% | 0.871 | 0.921 | ±0.022 |
| Logistic Regression | 84.6% | 0.839 | 0.901 | ±0.026 |

> Best model: **XGBoost** with tuned `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`

---

## 🔁 Pipeline

```
Raw Data (CSV/JSON)
      │
      ▼
┌─────────────────┐
│  Data Ingestion │  ← Schema validation, null checks, logging
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Preprocessing       │  ← Imputation, outlier removal, encoding, scaling
└──────────┬───────────┘
           │
           ▼
┌──────────────────┐
│  EDA             │  ← Correlation matrix, class balance, distributions
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Feature Engineering │  ← Composite scores, interactions, RFECV selection
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Model Training & CV Tuning      │  ← 5-fold stratified CV, GridSearchCV
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────┐
│  Evaluation & SHAP       │  ← Accuracy, F1, AUC, confusion matrix
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────┐
│  Flask REST API      │  ← /predict, /explain, /batch, /health
└──────────────────────┘
```

---

## 🗂 Project Structure

```
alzheimer-detection/
│
├── app/
│   ├── __init__.py
│   ├── routes.py            # Flask API endpoints
│   ├── predict.py           # Inference logic
│   └── schemas.py           # Request/response validation (Marshmallow)
│
├── pipeline/
│   ├── ingest.py            # Data loading & schema checks
│   ├── preprocess.py        # Cleaning, imputation, encoding, scaling
│   ├── eda.py               # Automated EDA report generation
│   ├── features.py          # Feature engineering & RFECV selection
│   ├── train.py             # Cross-validation, GridSearchCV, training
│   └── evaluate.py          # Metrics, confusion matrix, SHAP values
│
├── models/
│   └── xgboost_v1.pkl       # Serialised best model
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis
│   ├── 02_feature_eng.ipynb # Feature engineering experiments
│   └── 03_modelling.ipynb   # Model comparison & tuning
│
├── data/
│   ├── raw/                 # Original dataset (not committed)
│   └── processed/           # Cleaned, engineered features
│
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
│
├── Dockerfile
├── requirements.txt
├── config.py
├── run.py                   # Entry point
└── README.md
```

---

## 🔌 API Reference

### `POST /api/v1/predict`

Run classification on a single patient's clinical features.

**Request body:**

```json
{
  "age": 71,
  "mmse_score": 22,
  "cdr": 1.0,
  "apoe_e4": 1,
  "hippocampal_volume": 2.8,
  "cortical_thickness": 3.2,
  "education_years": 14,
  "gender": "M"
}
```

**Response:**

```json
{
  "prediction": "MCI",
  "risk_score": 0.724,
  "confidence": 0.891,
  "label": "Mild Cognitive Impairment",
  "model_version": "xgboost_v1"
}
```

---

### `GET /api/v1/explain`

Returns SHAP feature importance scores for the most recent prediction.

```json
{
  "feature_importances": {
    "mmse_score": 0.32,
    "cdr": 0.28,
    "hippocampal_volume": 0.21,
    "apoe_e4": 0.12,
    "age": 0.07
  }
}
```

---

### `POST /api/v1/batch`

Submit multiple patient records for batch inference.

```json
{
  "patients": [ { "age": 68, "mmse_score": 26, "..." : "..." }, { "..." : "..." } ]
}
```

---

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model": "xgboost_v1",
  "uptime_seconds": 3821
}
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/alzheimer-detection.git
cd alzheimer-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python pipeline/ingest.py      # Load & validate data
python pipeline/preprocess.py  # Clean & transform
python pipeline/eda.py         # Generate EDA report
python pipeline/features.py    # Feature engineering
python pipeline/train.py       # Train & tune model
python pipeline/evaluate.py    # Evaluate & save model
```

### 5. Start the Flask API

```bash
python run.py
# API running at http://localhost:5000
```

### 6. Test a prediction

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 71, "mmse_score": 22, "cdr": 1.0, "apoe_e4": 1}'
```

---

### Docker

```bash
docker build -t alzheimer-api .
docker run -p 5000:5000 alzheimer-api
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔑 Key Technical Decisions

**Why XGBoost?** Gradient boosted trees outperform linear models on tabular clinical data with mixed feature types. They handle missing values natively and are interpretable via SHAP values — critical in a medical context.

**Why stratified CV?** Class imbalance in Alzheimer's datasets (more healthy controls than MCI/AD cases) means random splits can inflate accuracy. Stratified folds preserve class proportions across every fold.

**Why RFECV for feature selection?** Recursive feature elimination with cross-validation finds the optimal feature subset without data leakage — unlike selecting features on the full dataset first.

**Why SHAP for explainability?** In clinical applications, a prediction without an explanation has limited utility. SHAP provides consistent, locally accurate feature attributions that a clinician can reason about.

---

## 📦 Dependencies

```
flask==2.3.3
scikit-learn==1.3.0
xgboost==1.7.6
pandas==2.0.3
numpy==1.24.4
shap==0.42.1
marshmallow==3.20.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
pytest==7.4.0
```

---

## ⚠️ Disclaimer

This project is a research and portfolio demonstration. It is **not** validated for clinical use and should not be used to make medical decisions. Always consult qualified medical professionals.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with Python · Flask · Scikit-learn · Pandas
</div>
