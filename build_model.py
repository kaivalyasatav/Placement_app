import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

print("🚀 Building fitted XGBoost placement model...")


df = pd.read_csv("student_training_dataset_realistic_perfect.csv")

X = df[["cgpa","avg_test_score","technical_score","aptitude_score",
        "num_projects","num_internships","branch"]]
y = df["placed"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ColumnTransformer
numeric = ["cgpa","avg_test_score","technical_score","aptitude_score","num_projects","num_internships"]
categorical = ["branch"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

# Pipeline
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Fit it (important!)
pipe.fit(X_train, y_train)

# Evaluate briefly
acc = pipe.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc:.3f}")

# Save the fitted pipeline
joblib.dump(pipe, "placement_model_xgb.joblib")
print("🎯 Saved as placement_model_xgb.joblib — fully fitted & ready to use.")