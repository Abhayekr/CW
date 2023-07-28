#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures


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
    Predict the number of tubes required based on user input using the trained model.
    Returns the predicted number of tubes.
    """
    user_input = [[dia, charge, density]]
    predicted_tubes = float(model.predict(user_input))

    return predicted_tubes

def get_float_input(prompt):
    """
    Prompt the user for a float input and handle exceptions for invalid inputs.
    Returns the valid float value entered by the user.
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid numeric value.")

def main():
    """
    Main function to run the Mould Tube Estimator.
    """

    # Load the dataset
    url = "https://raw.githubusercontent.com/Abhayekr/CW/main/Charge%20Weight10.csv"
    X, y = load_data(url)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model with hyperparameter optimization
    mlr_model = train_model(X_train, y_train)

    # Predict output variables for the test data
    y_pred = mlr_model.predict(X_test)

    # Calculate R-squared for each output variable using the test data
    r2_CW = r2_score(y_test, y_pred)

    # Calculate mean squared error for each output variable using the test data
    mse_CW = mean_squared_error(y_test, y_pred)

    print("MOULD TUBE ESTIMATOR\n")

    while True:
        # Prompt user for input
        print("\nPlease enter the details below: \n")
        dia = get_float_input("Tube Diameter (in mm): ")
        charge = get_float_input("Charge Weight (in kg): ")
        density = get_float_input("JMAT Pro Density: ")

        # Perform prediction
        predicted_tubes = predict_tubes(mlr_model, dia, charge, density)

        print("\nNumber of Tubes Required: ", predicted_tubes)

        # Check if user wants to calculate for another set of inputs
        choice = input("\nDo you want to calculate for another set of inputs? (Y/N): ")
        if choice.lower() != 'y':
            break

    # Print evaluation metrics
    print("\nEvaluation Metrics for No. of Tubes:")
    print("R-squared (R2):", r2_CW)
    print("Mean Squared Error (MSE):", mse_CW)

# Run the main function
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




