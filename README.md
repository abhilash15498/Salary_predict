# Salary_predict
I have built a web application which predicts salaries using linear regression .
# 💼 AI Salary Predictor

> Predict your annual salary based on your professional profile using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![R²](https://img.shields.io/badge/R²-0.969-brightgreen)

---

## About the project

A Streamlit web app that uses a trained **Linear Regression** model to estimate your annual salary (in LPA) based on 5 features: years of experience, education level, skills score, job role, and certifications.

Built as a practical ML mini-project to understand the full **data → model → UI** pipeline.

| Metric | Value |
|--------|-------|
| R² Score | 0.969 |
| RMSE | ~2.0 LPA |
| Training samples | 300 |

---

## Features

- Real-time salary prediction from 5 profile inputs
- Salary tier classification (Fresher → Top Tier)
- Personalized tips to improve your salary estimate
- Clean dark-themed Streamlit UI with custom CSS
- Sanity-checked model — freshers correctly get 3–5 LPA

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit |
| ML Model | scikit-learn · Linear Regression |
| Data | NumPy · Pandas |
| Model persistence | joblib |

---

## Project structure
```
salary_predictor/
├── app.py                 ← Streamlit UI
├── train_model.py         ← Model training + evaluation
├── generate_dataset.py    ← Synthetic dataset generation
├── salary_data.csv        ← Training data (300 rows)
├── salary_model.pkl       ← Saved model
└── requirements.txt
```

---

## Quick start
```bash
git clone https://github.com/yourusername/salary-predictor.git
cd salary-predictor
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

---

## Model details

Algorithm: **Linear Regression**

Each feature contributes independently:
- +1.1 LPA per year of experience
- +5 LPA for Master's over High School
- +0.07 LPA per skills score point
- +3.9 LPA per role level up
- +0.5 LPA per certification

> Trained on synthetic India IT market data. For learning purposes only.

---

## What I learned

- How Linear Regression learns coefficients from data
- Why training data range matters — model saw skills 30–100, so entering 0 gave wrong output (fixed in v2)
- What R² and RMSE actually mean in practice
- How to build an end-to-end ML pipeline: data → train → save → UI
- How Streamlit reruns the entire script on every user interaction
