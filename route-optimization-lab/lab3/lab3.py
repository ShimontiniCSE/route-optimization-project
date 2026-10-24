import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path Resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "Urban_Logistics_Node_Path_Dataset.csv")
df = pd.read_csv(file_path)

print("--- LAB 3: DATA VISUALIZATION & EDA ---")

# 1. Distribution Plots (Histograms & KDE)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['path_length'], kde=True, ax=axes[0], color='royalblue')
axes[0].set_title('Distribution of Path Length')

sns.histplot(df['delivery_cost'], kde=True, ax=axes[1], color='seagreen')
axes[1].set_title('Distribution of Delivery Cost')
plt.tight_layout()
plt.show()

# 2. Relationship Analysis (Scatter Plot & Box Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='travel_time', y='delivery_cost', hue='target_route_selection')
plt.title('Travel Time vs Delivery Cost by Target Route Selection')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='traffic_density', y='congestion_probability', palette='Set2')
plt.title('Congestion Probability by Traffic Density')
plt.show()

# 3. Correlation Matrix Heatmap
plt.figure(figsize=(10, 6))
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Route Metrics Correlation Matrix')
plt.show()