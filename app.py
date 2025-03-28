#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('fraud_model.pkl', 'rb'))  # Load your trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        amount = float(request.form['amount'])
        old_balance = float(request.form['oldbalanceOrg'])
        new_balance = float(request.form['newbalanceOrg'])
        transaction_type = request.form['transaction_type']
        
        # Convert transaction type to numerical value
        transaction_mapping = {"TRANSFER": 1, "CASH_OUT": 0}
        transaction_encoded = transaction_mapping.get(transaction_type, -1)
        
        if transaction_encoded == -1:
            return jsonify({'error': 'Invalid transaction type'})

        # Make prediction
        features = np.array([[amount, old_balance, new_balance, transaction_encoded]])
        prediction = model.predict(features)[0]

        # Convert to readable output
        result = "Fraudulent" if prediction == 1 else "Not Fraudulent"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


# In[ ]:




