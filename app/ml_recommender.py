import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/gym_data.csv")

# Features used for ML
features = ["Sex", "Age", "BMI", "Fitness Goal", "Level", "Hypertension", "Diabetes"]

# --- EXERCISE MODEL ---
encoders_ex = {}
for col in ["Sex", "Fitness Goal", "Level", "Hypertension", "Diabetes"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders_ex[col] = le
...
joblib.dump(encoders_ex, "app/exercise_encoders.pkl")

# --- DIET MODEL ---
encoders_diet = {}
for col in ["Sex", "Fitness Goal", "Level", "Hypertension", "Diabetes"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders_diet[col] = le
...
joblib.dump(encoders_diet, "app/diet_encoders.pkl")  # separate encoder
