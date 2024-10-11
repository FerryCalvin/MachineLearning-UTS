from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and preprocess
df = pd.read_csv('prostate_cancer_transformed.csv')

# Features and labels
X = df.drop(columns=['diagnosis_result'])
y = df['diagnosis_result']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model training
nb = GaussianNB()
nb.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Define a function for prediction
def make_prediction(input_data):
    # Transform input data using the same scaler
    input_data_scaled = scaler.transform([input_data])
    
    # Predictions from each model
    pred_nb = nb.predict(input_data_scaled)[0]
    pred_dt = dt.predict(input_data_scaled)[0]
    pred_rf = rf.predict(input_data_scaled)[0]
    
    return {
        'Naive Bayes': pred_nb,
        'Decision Tree': pred_dt,
        'Random Forest': pred_rf
    }

# Flask route for home page
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/index')
def diagnose():
    return render_template('index.html')

# Flask route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        radius = float(request.form['radius'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])
        area = float(request.form['area'])
        smoothness = float(request.form['smoothness'])
        compactness = float(request.form['compactness'])
        symmetry = float(request.form['symmetry'])
        fractal_dimension = float(request.form['fractal_dimension'])
        
        # Create input data list
        input_data = [radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension]
        
        # Get predictions
        predictions = make_prediction(input_data)
        
        # Render results back to HTML
        return render_template('index.html', 
                               pred_nb=predictions['Naive Bayes'],
                               pred_dt=predictions['Decision Tree'],
                               pred_rf=predictions['Random Forest'])

if __name__ == '__main__':
    app.run(debug=True)
