# ============================================
# app.py — AI Driven Placement Analytics Portal (Final Version)
# ============================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="AI Placement Analytics", layout="wide")

MODEL_PATH = "placement_model_xgb.joblib"
DATA_PATH = "student_training_dataset_realistic_perfect.csv"

# ============================================
# LOAD MODEL & DATASET
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully and verified.")
    return model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# ============================================
# SIDEBAR INPUTS
# ============================================
st.title("🎓 AI-Driven Placement Prediction & Analytics Dashboard")
st.caption("Get your placement probability, analytics insights, and detailed career recommendations.")

st.sidebar.header("Enter Student Information")

default = df.sample(1, random_state=42).iloc[0]
cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, float(default.cgpa), 0.01)
avg_test_score = st.sidebar.slider("Average Test Score", 40.0, 95.0, float(default.avg_test_score), 0.1)
technical_score = st.sidebar.slider("Technical Score", 35.0, 100.0, float(default.technical_score), 0.1)
aptitude_score = st.sidebar.slider("Aptitude Score", 35.0, 95.0, float(default.aptitude_score), 0.1)
num_projects = st.sidebar.slider("Number of Projects", 0, 6, int(default.num_projects))
num_internships = st.sidebar.slider("Number of Internships", 0, 3, int(default.num_internships))
branch = st.sidebar.selectbox("Branch", sorted(df.branch.unique()))

skills_input = st.sidebar.text_input("Skills (comma separated)", value=default.skills)
user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]

# ============================================
# PREDICTION
# ============================================
if st.sidebar.button("🔮 Predict Placement"):
    X = pd.DataFrame([{
        "cgpa": cgpa,
        "avg_test_score": avg_test_score,
        "technical_score": technical_score,
        "aptitude_score": aptitude_score,
        "num_projects": num_projects,
        "num_internships": num_internships,
        "branch": branch
    }])

    try:
        proba = model.predict_proba(X)[0][1]
        st.subheader("📈 Placement Prediction Result")
        st.metric("Placement Probability", f"{proba:.2%}")
        if proba > 0.5:
            st.success("✅ The student is likely to be placed!")
        else:
            st.warning("⚠️ The student is unlikely to be placed.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # ============================================
    # STUDENT ANALYTICS SECTION
    # ============================================
    st.divider()
    st.subheader("📊 Student Performance Analytics")

    mean_scores = df[["cgpa","avg_test_score","technical_score","aptitude_score"]].mean()

    cols = st.columns(4)
    cols[0].metric("Your CGPA", f"{cgpa:.2f}", delta=f"{cgpa - mean_scores['cgpa']:.2f} vs avg")
    cols[1].metric("Your Test Avg", f"{avg_test_score:.1f}", delta=f"{avg_test_score - mean_scores['avg_test_score']:.1f} vs avg")
    cols[2].metric("Technical Score", f"{technical_score:.1f}", delta=f"{technical_score - mean_scores['technical_score']:.1f}")
    cols[3].metric("Aptitude", f"{aptitude_score:.1f}", delta=f"{aptitude_score - mean_scores['aptitude_score']:.1f}")

    # Radar chart for profile visualization
    categories = ['CGPA','Technical','Aptitude','Projects','Internships']
    values = [cgpa/10, technical_score/100, aptitude_score/100, num_projects/6, num_internships/3]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name='You', line_color='blue'))
    fig.add_trace(go.Scatterpolar(
        r=[mean_scores['cgpa']/10, mean_scores['technical_score']/100,
           mean_scores['aptitude_score']/100, df['num_projects'].mean()/6,
           df['num_internships'].mean()/3],
        theta=categories, fill='toself', name='Batch Avg', line_color='orange'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # COMPANY-WISE PLACEMENT PREDICTIONS (Dynamic)
    # ============================================
    st.divider()
    st.subheader("🏢 Company-Wise Placement Predictions (Dataset + Top MNCs)")

    from collections import Counter
    # Use dataset companies directly
    dataset_companies = sorted([c for c in df["company"].dropna().unique() if isinstance(c, str) and c.strip() != ""])
    all_companies = dataset_companies

    # Calculate profiles dynamically
    company_profiles = {}
    placed_df = df[df['placed'] == 1].dropna(subset=['company'])
    for comp in all_companies:
        comp_data = placed_df[placed_df['company'] == comp]
        if len(comp_data) == 0:
            continue
            
        avg_cgpa = comp_data['cgpa'].mean()
        avg_tech = comp_data['technical_score'].mean()
        
        all_skills = []
        for s in comp_data['skills'].dropna():
            all_skills.extend([str(x).strip() for x in str(s).split(',')])
        top_skills = [k for k, v in Counter(all_skills).most_common(5)]
        if not top_skills:
            top_skills = ["Python", "SQL"]
            
        avg_sal = comp_data['salary'].mean() if 'salary' in comp_data.columns else 0
        tier = 3
        if avg_sal >= 1200000:
            tier = 1
        elif avg_sal >= 600000:
            tier = 2
            
        company_profiles[comp] = {
            "skills": top_skills,
            "cgpa": avg_cgpa,
            "tech": avg_tech,
            "tier": tier
        }

    company_probs = []
    for comp_name in all_companies:
        profile = company_profiles.get(comp_name, {
            "skills":["Python","SQL"], "cgpa":7.0, "tech":65, "tier":3
        })
        skill_match = len(set(profile["skills"]) & set(user_skills)) / max(len(profile["skills"]), 1)
        
        cgpa_req = max(profile.get("cgpa", 7.0), 1)
        tech_req = max(profile.get("tech", 65), 1)
        tier_val = profile.get("tier", 3)
        
        if tier_val == 1 and (cgpa < 8.0 or technical_score < 75):
            comp_prob = 0.0
        elif tier_val == 2 and (cgpa < 7.0 or technical_score < 65):
            comp_prob = 0.0
        else:
            cgpa_score = min(cgpa / cgpa_req, 1.2)
            tech_score = min(technical_score / tech_req, 1.2)
            # Tier 3 (easier) has highest normal probability. Top tiers are naturally harder unless you far exceed requirements.
            base_weight = {1: 0.7, 2: 0.85, 3: 1.0}.get(tier_val, 1.0)
            comp_prob = proba * (0.5 * skill_match + 0.25 * cgpa_score + 0.25 * tech_score) * base_weight
            comp_prob = min(comp_prob, 0.99)
            
        company_probs.append((comp_name, round(comp_prob*100, 2)))

    comp_df = pd.DataFrame(company_probs, columns=["Company","Probability (%)"]).sort_values("Probability (%)", ascending=False)
    st.bar_chart(comp_df.set_index("Company"))

    top5 = comp_df.head(5)
    st.success(f"🏆 Based on your profile, your best-fit companies are: **{', '.join(top5['Company'].tolist())}**")

    # ============================================
    # ENHANCED CAREER RECOMMENDATIONS SECTION
    # ============================================
    st.divider()
    st.subheader("💡 Comprehensive Career Recommendations")

    recommendations = []

    # 🎓 Academic Readiness
    if cgpa < 6.5:
        recommendations.append(("🎓 Academic Readiness", 
            "Your CGPA is below most company eligibility cutoffs. "
            "Focus on improving academics through reattempts, internal assessments, and consistent study habits. Aim for ≥ 7.5."))
    elif 6.5 <= cgpa < 7.5:
        recommendations.append(("🎓 Academic Readiness", 
            "Good academic standing. Maintain consistency and try to push above 7.5 for Tier 1 eligibility."))
    else:
        recommendations.append(("🎓 Academic Readiness", 
            "Excellent academics — eligible for Tier 1 companies. Maintain this record."))

    # 💻 Technical & Projects
    if technical_score < 60:
        recommendations.append(("💻 Technical Skills", 
            "Improve technical concepts — focus on DSA, OOPs, and hands-on coding. Use platforms like LeetCode, CodeChef, and HackerRank."))
    elif 60 <= technical_score < 75:
        recommendations.append(("💻 Technical Skills", 
            "You're doing well technically. Work on 1–2 impactful projects showcasing modern tech like React, Django, or ML."))
    else:
        recommendations.append(("💻 Technical Skills", 
            "Strong technical base! Explore advanced domains like Cloud, AI/ML, or Full-Stack Development."))

    # 🧮 Aptitude & Reasoning
    if aptitude_score < 60:
        recommendations.append(("🧮 Aptitude & Reasoning", 
            "Work on aptitude and reasoning — practice from PrepInsta or IndiaBix. Focus on time management."))
    elif 60 <= aptitude_score < 75:
        recommendations.append(("🧮 Aptitude & Reasoning", 
            "Good aptitude. Regular practice with mock tests can push your score higher."))
    else:
        recommendations.append(("🧮 Aptitude & Reasoning", 
            "Excellent aptitude — ideal for Tier 1 company assessments."))

    # 🧠 Projects & Internships
    if num_projects < 2:
        recommendations.append(("🧠 Projects & Internships", 
            "Build at least 2–3 projects — web apps, ML models, or dashboards. Showcase them on GitHub."))
    elif 2 <= num_projects < 4:
        recommendations.append(("🧠 Projects & Internships", 
            "Good progress! Add one strong, domain-relevant project (AI, Full Stack, or Data Analytics)."))
    else:
        recommendations.append(("🧠 Projects & Internships", 
            "Impressive project portfolio! Try hackathons or open-source contributions."))

    # 🌐 Communication & Personality
    if num_internships == 0:
        recommendations.append(("🌐 Communication & Personality", 
            "Participate in group activities, internships, and presentation events to build confidence and soft skills."))
    else:
        recommendations.append(("🌐 Communication & Personality", 
            "Strong exposure through internships — continue enhancing leadership and communication."))

    # 🏢 Company Fit Insights
    st.subheader("🏢 Company Fit Insights")
    for idx, row in comp_df.head(5).iterrows():
        if row["Probability (%)"] >= 80:
            st.success(f"✅ **{row['Company']}** — Strong Fit ({row['Probability (%)']}%)")
        elif 50 <= row["Probability (%)"] < 80:
            st.warning(f"🟡 **{row['Company']}** — Moderate Fit ({row['Probability (%)']}%)")
        else:
            st.info(f"⚪ **{row['Company']}** — Low Fit ({row['Probability (%)']}%)")

    # 📈 Next Steps
    recommendations.append(("📈 Next Steps", 
        "Enroll in specialized courses based on your career path. Examples:\n"
        "- **Amazon/Google** → Machine Learning, DSA, System Design\n"
        "- **Accenture/Capgemini** → Web & Cloud Technologies\n"
        "- **Deloitte** → Business Analytics & Python for Data Science."))

    # 🧭 Career Strategy
    recommendations.append(("🧭 Career Strategy", 
        "Maintain an active LinkedIn profile and GitHub repository. Post projects, connect with alumni, and apply for internships early."))

    # Display Recommendations
    for cat, msg in recommendations:
        st.markdown(f"### {cat}")
        st.write(msg)
        st.markdown("---")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("© 2025 Campus AI Placement Portal — Developed by Kaivalya Satav 💻")