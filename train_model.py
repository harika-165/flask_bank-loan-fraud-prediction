#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("Fraud.csv")

# Select relevant columns
selected_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"]
df = df[selected_columns].dropna()

# Convert categorical column 'type' to numerical values
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Ensure all expected columns exist
expected_features = ["amount", "oldbalanceOrg", "newbalanceOrig", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
for col in expected_features:
    if col not in df.columns:
        df[col] = 0  # Add missing columns with default 0

# Define features and target variable
X = df[expected_features]  # Keep only these features
y = df["isFraud"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model and expected feature order
joblib.dump({"model": pipeline, "features": expected_features}, "fraud_model.pkl")

print("Model training complete. Saved as fraud_model.pkl.")


# In[ ]:




