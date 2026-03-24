import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Dataset_models/fatty liver.csv")

# Clean dataset
df = df.drop(columns=["Unnamed: 0", "id", "case.id", "futime"])

# Features + target
X = df[["age", "male", "bmi"]]
y = df["status"]

# Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

# Save model
joblib.dump(pipeline, "Models/fatty_liver_model.joblib")
print("✅ Fatty Liver model retrained successfully")
