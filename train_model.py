import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ── Load ─────────────────────────────────────
df = pd.read_csv("salary_data.csv")
print(f"Dataset loaded: {df.shape}")

# ── Features & target ─────────────────────────
X = df[["experience", "education", "skills_score", "job_role", "certifications"]]
y = df["salary_lpa"]

# ── Train / test split ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train model ───────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────
y_pred  = model.predict(X_test)
mse     = mean_squared_error(y_test, y_pred)
r2      = r2_score(y_test, y_pred)

print(f"\nR² Score : {r2:.4f}")
print(f"RMSE     : {np.sqrt(mse):.4f} LPA")
print(f"\nCoefficients:")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:20s}: {coef:+.4f}")
print(f"  {'intercept':20s}: {model.intercept_:+.4f}")

# ── Sanity check predictions ──────────────────
test_cases = [
    {"label": "True fresher (0 exp, HS, 20 skills, Analyst, 0 certs)",
     "vals": [0, 1, 20, 1, 0]},
    {"label": "Fresh grad (0 exp, Bachelor, 60 skills, Engineer, 1 cert)",
     "vals": [0, 2, 60, 2, 1]},
    {"label": "Mid-level (5 yrs, Bachelor, 70 skills, Engineer, 2 certs)",
     "vals": [5, 2, 70, 2, 2]},
    {"label": "Senior (12 yrs, Master, 85 skills, Senior, 4 certs)",
     "vals": [12, 3, 85, 3, 4]},
    {"label": "Director (20 yrs, PhD, 95 skills, Manager, 5 certs)",
     "vals": [20, 4, 95, 4, 5]},
]

print("\n── Sanity check predictions ──")
for case in test_cases:
    pred = model.predict([case["vals"]])[0]
    pred = max(2.5, pred)
    print(f"  {case['label']}")
    print(f"    → Predicted: ₹{pred:.2f} LPA\n")

# ── Save ─────────────────────────────────────
joblib.dump(model, "salary_model.pkl")
print("Model saved -> salary_model.pkl")
