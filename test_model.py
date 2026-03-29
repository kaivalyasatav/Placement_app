import pandas as pd
import joblib

# Load the trained model
print("Loading model...")
model = joblib.load("placement_model_xgb.joblib")

# 1. Very High Profile (Should confidently predict placement & tier 1 companies)
high_profile = pd.DataFrame([{
    "cgpa": 9.2,
    "avg_test_score": 90.0,
    "technical_score": 95.0,
    "aptitude_score": 90.0,
    "num_projects": 4,
    "num_internships": 2,
    "branch": "CSE"
}])

# 2. Average Profile 
avg_profile = pd.DataFrame([{
    "cgpa": 7.0,
    "avg_test_score": 70.0,
    "technical_score": 65.0,
    "aptitude_score": 60.0,
    "num_projects": 1,
    "num_internships": 0,
    "branch": "MECH"
}])

# 3. Very Low Profile (Should predict not placed)
low_profile = pd.DataFrame([{
    "cgpa": 5.2,
    "avg_test_score": 45.0,
    "technical_score": 40.0,
    "aptitude_score": 40.0,
    "num_projects": 0,
    "num_internships": 0,
    "branch": "CIVIL"
}])


profiles = [("High Profile", high_profile), ("Average Profile", avg_profile), ("Low Profile", low_profile)]

print("\n--- Model Predictions ---")
for name, data in profiles:
    # Predict Probability
    prob = model.predict_proba(data)[0][1]
    is_placed = prob > 0.5
    status = "✅ PLACED" if is_placed else "❌ NOT PLACED"
    print(f"{name}: {status} (Probability: {prob:.2%})")
