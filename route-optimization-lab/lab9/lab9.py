import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter GUI backend errors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Urban_Logistics_Node_Path_Dataset.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Select numeric features for route clustering
X = df.select_dtypes(include=['number'])

print("Calculating Within-Cluster Sum of Squares (WCSS) for K=1 to 10...")
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)
plot_path = os.path.join(plots_dir, "route_elbow_plot.png")

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--', color='b')
plt.title("Lab 9: Elbow Method for Route Clustering")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.savefig(plot_path)
plt.close()

# Fit optimal K-Means model (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Route_Cluster'] = kmeans.fit_predict(X)

print("\n=== Lab 9: K-Means Route Clustering Completed ===")
print("Cluster Distribution:")
print(df['Route_Cluster'].value_counts())
print(f"\nSaved elbow plot to: {plot_path}")