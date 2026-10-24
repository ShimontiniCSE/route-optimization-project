import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Path Resolution & Loading Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "Urban_Logistics_Node_Path_Dataset.csv")
df = pd.read_csv(file_path)

print("--- LAB 2: DATA PREPROCESSING & EXPLORATION ---")

# 2. Handling Duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows.")

# 3. Handling Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

# 4. Descriptive Statistics
print("\nDescriptive Statistics (Numerical Attributes):")
print(df.describe())

# 5. Encoding Categorical Variables (Updated for Pandas compatibility)
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
if categorical_cols:
    print(f"\nCategorical Columns to Encode: {categorical_cols}")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    df_encoded = df.copy()

# 6. Feature Scaling (Standardization)
numerical_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Exclude node IDs from scaling as they are key identifiers, not continuous metrics
ids_to_exclude = ['source_node_id', 'destination_node_id']
scale_features = [col for col in numerical_cols if col not in ids_to_exclude]

scaler = StandardScaler()
df_encoded[scale_features] = scaler.fit_transform(df_encoded[scale_features])

print("\nPreprocessed & Scaled Sample Data:")
print(df_encoded.head())