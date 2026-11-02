import joblib
from xgboost import XGBClassifier

# -----------------------------------
# 1. Load Preprocessor
# -----------------------------------
print("Loading preprocessor...")

preprocessor = joblib.load("CSAT_Preprocessor.joblib")


# -----------------------------------
# 2. Load XGBoost Model
# -----------------------------------
print("Loading XGBoost model...")

model = XGBClassifier()
model.load_model("CSAT_XGBoost.json")


# -----------------------------------
# 3. Display Model Classes
# -----------------------------------
print("\nModel Classes:", model.classes_)


# -----------------------------------
# 4. CSAT Score Mapping
# -----------------------------------
csat_mapping = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5
}


# -----------------------------------
# 5. Display Mapping
# -----------------------------------
print("\nCSAT Score Mapping:")

for encoded_class, csat_score in csat_mapping.items():
    print(f"Class {encoded_class} -> CSAT Score {csat_score}")


# -----------------------------------
# 6. Loading Status
# -----------------------------------
print("\nModel and Preprocessor Loaded Successfully!")