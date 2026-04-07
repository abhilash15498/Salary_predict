import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    .title-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1.5rem;
    }

    .result-card {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: #1a1a2e;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(247,151,30,0.4);
    }

    .tier-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }

    label { color: #e0e0e0 !important; font-weight: 600; }

    .stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="title-card">
    <h1 style="color:#ffffff; font-size:2.2rem; margin:0;">💼 AI Salary Predictor</h1>
    <p style="color:#ffd200; margin-top:0.5rem; font-size:1rem;">
        Fill in your profile and discover your estimated annual salary (India IT market).
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
MODEL_PATH = "salary_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Run train_model.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Input section ────────────────────────────────────────────
st.markdown("### 🧑‍💻 Your Professional Profile")

col1, col2 = st.columns(2)

with col1:
    experience = st.slider(
        "🗓️ Years of Experience",
        min_value=0, max_value=30, value=0, step=1,  # FIX: starts at 0
        help="Total professional work experience"
    )
    education = st.selectbox(
        "🎓 Highest Education",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "High School / Diploma",
            2: "Bachelor's Degree",
            3: "Master's Degree",
            4: "PhD / Doctorate"
        }[x],
        index=0,   # FIX: defaults to High School (most honest default)
        help="Your highest completed qualification"
    )
    certifications = st.slider(
        "📜 Number of Certifications",
        min_value=0, max_value=10, value=0, step=1,  # FIX: starts at 0
        help="Professional certifications (AWS, GCP, PMP, etc.)"
    )

with col2:
    skills_score = st.slider(
        "🛠️ Skills Score",
        min_value=0, max_value=100, value=30, step=1,  # FIX: starts at 0
        help="Rate your overall technical + soft skills honestly (0 = none, 100 = expert)"
    )
    job_role = st.selectbox(
        "🏢 Job Role Level",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Analyst / Fresher",
            2: "Engineer / Developer",
            3: "Senior / Lead",
            4: "Manager / Architect"
        }[x],
        index=0,   # FIX: defaults to Analyst
        help="Your current or target role level"
    )

# ── Predict button ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("💰 Predict My Salary"):
    # Build input as a DataFrame with named columns — avoids sklearn feature name warning
    input_df = pd.DataFrame([[experience, education, skills_score, job_role, certifications]],
                             columns=["experience", "education", "skills_score",
                                      "job_role", "certifications"])

    raw_prediction = model.predict(input_df)[0]

    # FIX: Realistic floor — no one in India IT gets less than 2.5 LPA
    prediction = max(2.5, round(raw_prediction, 2))

    # ── Salary tier logic ─────────────────────────────────────
    if prediction >= 50:
        tier, emoji, badge_color = "Top Tier 🚀 (Senior/Leadership)", "🏆", "#6c63ff"
    elif prediction >= 30:
        tier, emoji, badge_color = "High Package 💎 (Mid-Senior)", "🌟", "#00b09b"
    elif prediction >= 15:
        tier, emoji, badge_color = "Good Package 👍 (Mid-Level)", "👍", "#f7971e"
    elif prediction >= 8:
        tier, emoji, badge_color = "Entry Package 📈 (Junior)", "📈", "#4facfe"
    elif prediction >= 4:
        tier, emoji, badge_color = "Fresher Package 🌱 (Entry)", "🌱", "#a18cd1"
    else:
        tier, emoji, badge_color = "Starting Out 🎓 (Intern level)", "🎓", "#888"

    st.markdown(f"""
    <div class="result-card">
        {emoji} &nbsp; Estimated Salary: <b>₹ {prediction:.2f} LPA</b><br>
        <span class="tier-badge" style="background:{badge_color}; color:white;">
            {tier}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Profile summary ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📌 Profile Summary")

    edu_map  = {1: "High School / Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
    role_map = {1: "Analyst/Fresher", 2: "Engineer/Dev", 3: "Senior/Lead", 4: "Manager/Architect"}

    summary = {
        "Profile Factor":   ["Experience", "Education", "Skills Score", "Job Role", "Certifications"],
        "Your Value": [
            f"{experience} year{'s' if experience != 1 else ''}",
            edu_map[education],
            f"{skills_score} / 100",
            role_map[job_role],
            f"{certifications} cert{'s' if certifications != 1 else ''}"
        ],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # ── Personalized advice ───────────────────────────────────
    tips = []
    if experience < 2:
        tips.append("🎯 **Focus on building projects** — real experience is the biggest salary driver.")
    if skills_score < 50:
        tips.append("📚 **Upskill actively** — improving your skills score from 30→70 can add ~3 LPA.")
    if certifications == 0:
        tips.append("📜 **Get 1–2 certifications** (AWS/GCP/Azure) — each adds ~0.5–1 LPA on average.")
    if education == 1:
        tips.append("🎓 **A Bachelor's degree** adds a meaningful bump in most IT companies.")

    if tips:
        st.markdown("#### 💡 Tips to increase your salary")
        for tip in tips:
            st.markdown(tip)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<hr style="border-color: rgba(255,255,255,0.1); margin-top:3rem;">
<p style="text-align:center; color:#6b8fa3; font-size:0.8rem;">
    Built with 🔥⚡ using Streamlit &amp; scikit-learn &nbsp;|&nbsp;
    Linear Regression · India IT Market · Synthetic training data
</p>
""", unsafe_allow_html=True)
