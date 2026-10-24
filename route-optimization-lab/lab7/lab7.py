import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Urban_Logistics_Node_Path_Dataset.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Target column from Lab 6
target_col = 'target_route_selection'

X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
}

print("\n=== Lab 7: Gaussian Naive Bayes Results ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab7_results.csv")
pd.DataFrame([metrics]).to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")