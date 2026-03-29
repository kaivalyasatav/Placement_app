from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

app = FastAPI(title="Placement Prediction API")

# Load model globally
MODEL_PATH = "placement_model_xgb.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# Dynamically load company profiles from dataset
company_profiles = {}
try:
    df = pd.read_csv("student_training_dataset_realistic_perfect.csv")
    placed_df = df[df['placed'] == 1].dropna(subset=['company'])
    
    from collections import Counter
    for comp in placed_df['company'].unique():
        if not isinstance(comp, str) or comp.strip() == "": continue
        comp_data = placed_df[placed_df['company'] == comp]
        avg_cgpa = comp_data['cgpa'].mean()
        avg_tech = comp_data['technical_score'].mean()
        
        all_skills = []
        for s in comp_data['skills'].dropna():
            all_skills.extend([str(x).strip().lower() for x in str(s).split(',')])
        top_skills = [k for k, v in Counter(all_skills).most_common(5)]
        
        avg_sal = comp_data['salary'].mean() if 'salary' in comp_data.columns else 0
        tier = 3
        if avg_sal >= 1200000: tier = 1
        elif avg_sal >= 600000: tier = 2
        
        company_profiles[comp] = {
            "skills": top_skills,
            "cgpa": avg_cgpa,
            "tech": avg_tech,
            "tier": tier
        }
    print(f"✅ Loaded profiles for {len(company_profiles)} companies from dataset.")
except Exception as e:
    print(f"❌ Error loading company profiles: {e}")

class StudentData(BaseModel):
    student_id: str
    name: str
    branch: str
    cgpa: float
    avg_test_score: float
    technical_score: float
    aptitude_score: float
    num_projects: int
    num_internships: int
    skills: List[str]

class PredictionResponse(BaseModel):
    placed: bool
    company: str
    probability: float
    company_probabilities: dict
    salary: Optional[float] = None

@app.post("/predict", response_model=PredictionResponse)
def predict_placement(student: StudentData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Format data for prediction (matches training dataset columns)
    X = pd.DataFrame([{
        "cgpa": student.cgpa,
        "avg_test_score": student.avg_test_score,
        "technical_score": student.technical_score,
        "aptitude_score": student.aptitude_score,
        "num_projects": student.num_projects,
        "num_internships": student.num_internships,
        "branch": student.branch
    }])

    # Predict Probability
    try:
        proba = model.predict_proba(X)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    placed = bool(proba > 0.5)

    best_company = "None"
    best_prob = 0.0
    company_probs = {}

    if placed:
        user_skills = [s.strip().lower() for s in student.skills]
        for comp_name, profile in company_profiles.items():
            profile_skills = list(profile["skills"]) # type: ignore
            skill_match = len(set([s.lower() for s in profile_skills]) & set(user_skills)) / max(len(profile_skills), 1)
            
            cgpa_req = max(float(profile["cgpa"]), 1) # type: ignore
            tech_req = max(float(profile["tech"]), 1) # type: ignore
            tier_val = int(profile["tier"]) # type: ignore
            
            # Enforce hard cutoffs for Tier 1 and Tier 2 companies
            if tier_val == 1 and (student.cgpa < 8.0 or student.technical_score < 75):
                comp_prob = 0.0
            elif tier_val == 2 and (student.cgpa < 7.0 or student.technical_score < 65):
                comp_prob = 0.0
            else:
                cgpa_score = min(student.cgpa / cgpa_req, 1.2)
                tech_score = min(student.technical_score / tech_req, 1.2)
                base_weight = {1: 0.7, 2: 0.85, 3: 1.0}.get(tier_val, 1.0)
                comp_prob = proba * (0.5*skill_match + 0.25*cgpa_score + 0.25*tech_score) * base_weight
                comp_prob = min(comp_prob, 0.99)
                
            company_probs[comp_name] = float(comp_prob)
            if comp_prob > best_prob:
                best_prob = comp_prob
                best_company = comp_name

    return {
        "placed": placed,
        "company": best_company,
        "probability": float(proba),
        "company_probabilities": company_probs,
        "salary": None # Optional, left as null as requested
    }
