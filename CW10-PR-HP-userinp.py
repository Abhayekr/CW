#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv("Charge Weight10.csv")

# Define input features (independent variables) and output variables (dependent variables)
X = data[["Tube Dia (mm)","Charge Weight (kg)5","JMAT Pro Density"]]  # Input features (independent variables)
y = data[["No. of Tubes6"]]  # Output variables (dependent variables)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Create an instance of the linear regression model
mlr_model = LinearRegression()

# Fit the model to the training data
mlr_model.fit(X_train_poly, y_train)

# Predict output variables for the test data
y_pred = mlr_model.predict(X_test_poly)

# Calculate R-squared for each output variable using the test data
r2_CW = r2_score(y_test["No. of Tubes6"], y_pred[:,0])

# Calculate mean squared error for each output variable using the test data
mse_CW = mean_squared_error(y_test["No. of Tubes6"], y_pred[:,0])


# Print evaluation metrics
print("R-squared for CW:", r2_CW)

print("MSE for CW:", mse_CW)

# Python program showing
# a use of input()

while True:

    print("Enter the Details below: \n")
    Dia = float(input("\nEnter the Tube Dia in mm: "))
    Charge = float(input("\nEnter the Charge Weight in kg: "))
    Density = float(input("\nEnter the JMAT Pro Density: "))

    predict_poly = poly.fit_transform([[Dia,Charge,Density]])

    usr_pred = float(mlr_model.predict(predict_poly))
    print("\n \nNo. of Tubes Required: ", usr_pred)


# In[ ]:




