# ==========================================================
# Weather Prediction using Decision Tree & Random Forest
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/seattle-weather.csv")

# ==========================================================
# Data Cleaning
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# ==========================================================
# Feature Engineering
# ==========================================================

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# Drop original date column
df.drop("date", axis=1, inplace=True)

# ==========================================================
# Label Encoding
# ==========================================================

label_encoder = LabelEncoder()

df["weather"] = label_encoder.fit_transform(df["weather"])

print("\nWeather Classes:")
print(label_encoder.classes_)

# ==========================================================
# Feature Selection
# ==========================================================

X = df.drop("weather", axis=1)

y = df["weather"]

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# Decision Tree Model
# ==========================================================

print("\n" + "=" * 60)
print("Training Decision Tree")
print("=" * 60)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

decision_tree.fit(X_train_scaled, y_train)

dt_prediction = decision_tree.predict(X_test_scaled)

dt_accuracy = accuracy_score(y_test, dt_prediction)

print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")

print("\nClassification Report")
print(classification_report(y_test, dt_prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_prediction))

# ==========================================================
# Random Forest Model
# ==========================================================

print("\n" + "=" * 60)
print("Training Random Forest")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=42
)

random_forest.fit(X_train_scaled, y_train)

rf_prediction = random_forest.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print(f"Random Forest Accuracy : {rf_accuracy:.4f}")

print("\nClassification Report")
print(classification_report(y_test, rf_prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_prediction))

# ==========================================================
# Model Comparison
# ==========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")
print(f"Random Forest Accuracy : {rf_accuracy:.4f}")

if rf_accuracy > dt_accuracy:
    print("\nBest Model : Random Forest")
elif dt_accuracy > rf_accuracy:
    print("\nBest Model : Decision Tree")
else:
    print("\nBoth models have the same accuracy.")

# ==========================================================
# Save Models
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(decision_tree, "models/decision_tree_model.pkl")
joblib.dump(random_forest, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\n" + "=" * 60)
print("Models Saved Successfully!")
print("=" * 60)

print("✔ decision_tree_model.pkl")
print("✔ random_forest_model.pkl")
print("✔ scaler.pkl")
print("✔ label_encoder.pkl")