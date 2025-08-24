# ============================================
# Loan Approval Predictor – FINAL (Scikit-learn)
# Works with Kaggle file: Loan Approval Dataset.csv
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- 1) Load ----------
CSV_PATH = "Week 5 Project GNCIPL/archive (11).zip"  
df = pd.read_csv(CSV_PATH)

# ---------- 2) Drop useless ID ----------
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

# ---------- 3) Split features/target ----------
TARGET = "Loan_Status"
X = df.drop(columns=[TARGET])
y = df[TARGET]         # 'Y' / 'N'

# ---------- 4) Column types ----------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# NOTE: 'Dependents' 

# ---------- 5) Preprocess (impute + one-hot) ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

# ---------- 6) Model ----------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[("preprocess", preprocess),
                      ("model", model)])

# ---------- 7) Train / Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 8) Fit ----------
pipe.fit(X_train, y_train)

# ---------- 9) Evaluate ----------
y_pred = pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ---------- 10) Top Feature Importances ----------
# map back feature names (num + onehot cat)
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([num_cols, cat_feature_names])

importances = pipe.named_steps["model"].feature_importances_
feat_imp = (pd.DataFrame({"feature": all_feature_names, "importance": importances})
              .sort_values("importance", ascending=False)
              .head(15))
print("\nTop Feature Importances:\n", feat_imp)

# ---------- 11) Predict on a new application ----------
def predict_one(sample_dict: dict):
    """
    sample_dict keys must match original column names in the CSV (except Loan_ID),
    e.g. Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
    CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
    """
    sample_df = pd.DataFrame([sample_dict])
    return pipe.predict(sample_df)[0]

example_app = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "2",          # "0","1","2","3+" allowed
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,      # 1.0 or 0.0
    "Property_Area": "Urban"    # "Urban","Rural","Semiurban"
}

print("\nSample Prediction (Y=Approved, N=Not Approved):",
      predict_one(example_app))
