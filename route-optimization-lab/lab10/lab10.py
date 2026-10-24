import os
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "routes_processed.csv")

df = pd.read_csv(data_path)
feature_cols = ['Distance_km', 'Traffic_Density', 'Num_Stops', 'Speed_Limit_kmh']
X = df[[c for c in feature_cols if c in df.columns]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
df['Agglo_Cluster'] = agg.fit_predict(X_scaled)

print("=== Lab 10: Density & Hierarchical Route Clustering ===")
print("DBSCAN Route Cluster Distribution (-1 represents bottleneck anomalies):")
print(df['DBSCAN_Cluster'].value_counts())

print("\nAgglomerative Route Cluster Distribution:")
print(df['Agglo_Cluster'].value_counts())