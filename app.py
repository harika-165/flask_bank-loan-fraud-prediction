#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Define valid transaction types (based on dataset)
valid_transaction_types = ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

@app.route('/')
def home():
    return render_template('index.html', transaction_types=valid_transaction_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        transaction_type = request.form['transaction_type']

        # One-hot encoding for transaction type
        input_data = {
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "type_CASH_OUT": 1 if transaction_type == "CASH_OUT" else 0,
            "type_DEBIT": 1 if transaction_type == "DEBIT" else 0,
            "type_PAYMENT": 1 if transaction_type == "PAYMENT" else 0,
            "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

        return render_template('index.html', prediction=result, transaction_types=valid_transaction_types)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", transaction_types=valid_transaction_types)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




