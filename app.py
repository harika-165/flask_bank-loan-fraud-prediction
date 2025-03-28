#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("fraud_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        amount = float(data['amount'])
        oldbalanceOrg = float(data['oldbalanceOrg'])
        newbalanceOrig = float(data['newbalanceOrig'])
        type_transfer = 1 if data['type'] == 'TRANSFER' else 0
        type_cash_out = 1 if data['type'] == 'CASH_OUT' else 0

        # Create input array
        features = np.array([[amount, oldbalanceOrg, newbalanceOrig, type_cash_out, type_transfer]])

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




