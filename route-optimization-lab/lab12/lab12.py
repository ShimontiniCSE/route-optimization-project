import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
data_path = os.path.join(BASE_DIR, "data", "Urban_Logistics_Node_Path_Dataset.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

def train_model():
    if not os.path.exists(data_path):
        return None, []
    df = pd.read_csv(data_path)
    target_col = 'target_route_selection'
    
    X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)

model, feature_names = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', feature_names=feature_names)
    
    if request.method == 'POST':
        try:
            form_data = {feat: float(request.form.get(feat, 0)) for feat in feature_names}
            input_df = pd.DataFrame([form_data])
            
            prediction = model.predict(input_df)[0]
            
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Error evaluating route selection: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)