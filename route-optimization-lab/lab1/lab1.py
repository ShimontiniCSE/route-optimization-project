import os
import pandas as pd

# 1. Path Resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "Urban_Logistics_Node_Path_Dataset.csv")

# 2. Reading CSV File using Pandas (Lab 1 requirement)
df = pd.read_csv(file_path)

# 3. Basic Exploration
print("--- LAB 1: DATASET INSPECTION ---")
print(f"Dataset Shape (Rows, Columns): {df.shape}")

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Summary Info:")
df.info()