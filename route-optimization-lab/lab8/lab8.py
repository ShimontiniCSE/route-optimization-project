import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Urban_Logistics_Node_Path_Dataset.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing dataset at: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Target column from previous labs
target_col = 'target_route_selection'

X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []
for kernel in ['linear', 'rbf']:
    model = SVC(kernel=kernel, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results.append({
        'Kernel': kernel,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, preds, average='weighted', zero_division=0)
    })

results_df = pd.DataFrame(results)
print("\n=== Lab 8: Support Vector Machine (SVM) Results ===")
print(results_df)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab8_svm_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")