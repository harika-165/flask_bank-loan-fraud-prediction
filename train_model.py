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

# Define features and target variable
X = df.drop(columns=["isFraud"])
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

# Save the trained model
joblib.dump(pipeline, "fraud_model.pkl")

print("Model training complete. Saved as fraud_model.pkl.")


# In[ ]:




