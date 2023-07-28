#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import sys


def load_data(url):
    """
    Load the dataset from the given URL.
    Returns X (input features) and y (output variables).
    """
    data = pd.read_csv(url)
    X = data[["Tube Dia (mm)", "Charge Weight (kg)5", "JMAT Pro Density"]]
    y = data[["No. of Tubes6"]]
    return X, y

def train_model(X, y):
    """
    Train a polynomial regression model with hyperparameter optimization.
    Returns the trained model.
    """
    pipe = make_pipeline(
        PolynomialFeatures(),
        LinearRegression()
    )

    # Define the parameter grid
    param_grid = {'polynomialfeatures__degree': [2, 3, 4, 5]}  # Vary the degree of the polynomial features

    # Perform grid search
    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model

def predict_tubes(model, dia, charge, density):
    """
    Predict the number of tubes required using the trained model and provided input.
    Returns the predicted number of tubes.
    """
    user_input = [[dia, charge, density]]
    predicted_tubes = float(model.predict(user_input))

    return predicted_tubes

def main(dia, charge, density):
    """
    Main function to run the Mould Tube Estimator using provided input.
    """

    # Load the dataset
    url = "https://raw.githubusercontent.com/Abhayekr/CW/main/Charge%20Weight10.csv"
    X, y = load_data(url)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model with hyperparameter optimization
    mlr_model = train_model(X_train, y_train)

    # Perform prediction
    predicted_tubes = predict_tubes(mlr_model, dia, charge, density)

    return predicted_tubes

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filename.py <Tube Diameter> <Charge Weight> <JMAT Pro Density>")
    else:
        dia = float(sys.argv[1])
        charge = float(sys.argv[2])
        density = float(sys.argv[3])
        result = main(dia, charge, density)
        print("Number of Tubes Required:", result)


# In[ ]:





# In[ ]:




