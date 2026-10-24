import os
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "routes_processed.csv")

df = pd.read_csv(data_path)
target_col = 'Delay_Risk'
X = df.drop(columns=['Route_ID', target_col], errors='ignore')
y = df[target_col]

# 1. Isolation Forest for Congestion/Outlier Route Detection
iso = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly_Flag'] = iso.fit_predict(X)

print("=== Lab 11: Route Anomaly Detection & Feature Importance ===")
print(f"Anomalous/Severe Delay Routes Detected: {sum(df['Anomaly_Flag'] == -1)}")

# 2. Random Forest Feature Importance Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRoute Factor Importances:")
print(importance_df)