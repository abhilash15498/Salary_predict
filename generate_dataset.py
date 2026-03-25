import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

# ── Features ──────────────────────────────────────────────────
# Skills NOW ranges 0–100 (so the model sees low skills too)
experience      = np.random.uniform(0, 30, n)
education       = np.random.randint(1, 5, n)   # 1=HighSchool 2=Bachelor 3=Master 4=PhD
skills_score    = np.random.uniform(0, 100, n)  # FIX: was 30-100, now 0-100
job_role        = np.random.randint(1, 5, n)    # 1=Analyst 2=Eng 3=Senior 4=Manager
certifications  = np.random.randint(0, 6, n)
noise           = np.random.normal(0, 1.5, n)

# ── Realistic salary formula ───────────────────────────────────
#
# Real-world logic:
#   - A fresher (0 exp, low skills, role=1) should get ~3–5 LPA
#   - A mid-level (5 yrs, decent skills, Bachelor) should get ~10–18 LPA
#   - Senior (15 yrs, high skills, Master, Manager) should get ~35–60 LPA
#
# education BONUS (not base):
#   HighSchool=0, Bachelor=2, Master=5, PhD=9
edu_bonus = np.where(education == 1, 0,
            np.where(education == 2, 2,
            np.where(education == 3, 5, 9)))

# role BONUS (not base):
#   Analyst=0, Engineer=2, Senior=6, Manager=12
role_bonus = np.where(job_role == 1, 0,
             np.where(job_role == 2, 2,
             np.where(job_role == 3, 6, 12)))

salary = (
    2.5                             # realistic base (minimum floor)
    + 1.1  * experience             # each year of exp = +1.1 LPA
    + edu_bonus                     # education bump (not a multiplier)
    + role_bonus                    # role level bump
    + 0.08 * skills_score           # 100 skills → +8 LPA max
    + 0.6  * certifications         # each cert → +0.6 LPA
    + noise
)

# Clip to realistic India IT market range: 2.5 to 80 LPA
salary = np.clip(salary, 2.5, 80).round(2)

df = pd.DataFrame({
    "experience":     experience.round(1),
    "education":      education,
    "skills_score":   skills_score.round(1),
    "job_role":       job_role,
    "certifications": certifications,
    "salary_lpa":     salary,
})

df.to_csv("salary_data.csv", index=False)
print("Dataset saved -> salary_data.csv")
print(df.describe().round(2))

# Sanity check: what does a true fresher look like?
print("\n── Fresher sanity check ──")
freshers = df[(df.experience < 1) & (df.skills_score < 40) & (df.job_role == 1)]
print(f"Freshers (0 exp, low skills, Analyst): avg salary = {freshers['salary_lpa'].mean():.2f} LPA")
print(freshers[["experience","education","skills_score","job_role","certifications","salary_lpa"]].head())
