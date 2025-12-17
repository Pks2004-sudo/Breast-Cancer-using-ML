import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Drop unwanted columns safely
data = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Convert target labels
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Select features (MUST match app.py inputs)
X = data[
    [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
    ]
]

y = data["diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ breast_cancer_model.pkl and scaler.pkl created successfully")
