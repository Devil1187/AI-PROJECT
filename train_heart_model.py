import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset 
df = pd.read_csv(
    r"C:\Users\HP\Desktop\CODING\Disease_Diagnosis-with-AI\Dataset_models\heart.csv"
)
  # 🔴 change name if needed

# 2. Split features and target
X = df.drop("target", axis=1)   # 🔴 target column name
y = df["target"]

# 3. Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 5. Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# 6. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]
)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train
pipeline.fit(X_train, y_train)

# 9. Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save model (SAFE WAY)
joblib.dump(pipeline, "Models/heart_model.joblib")

print("✅ Model retrained and saved successfully")
