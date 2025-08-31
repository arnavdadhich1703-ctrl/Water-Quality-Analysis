# water_quality_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Replace with your dataset file path
data = pd.read_csv("water_quality.csv")

print("🔹 Dataset Shape:", data.shape)
print("\n🔹 First 5 rows:\n", data.head())

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
print("\n🔹 Missing Values Before:\n", data.isnull().sum())

# Fill numerical missing values with median
for col in data.columns:
    if data[col].dtype in ["float64", "int64"]:
        data[col] = data[col].fillna(data[col].median())

print("\n🔹 Missing Values After:\n", data.isnull().sum())

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Water Quality Features")
plt.savefig("correlation_heatmap.png")
plt.close()

# Distribution of pH
plt.figure(figsize=(6, 4))
sns.histplot(data["ph"], kde=True, bins=30, color="blue")
plt.title("Distribution of pH")
plt.savefig("ph_distribution.png")
plt.close()

# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

data_scaled = data.copy()
data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print("\n🔹 Scaled Dataset Sample:\n", data_scaled.head())

# -----------------------------
# 5. Save Preprocessed Data
# -----------------------------
data_scaled.to_csv("water_quality_cleaned.csv", index=False)
print("\n✅ Preprocessed dataset saved as 'water_quality_cleaned.csv'")
