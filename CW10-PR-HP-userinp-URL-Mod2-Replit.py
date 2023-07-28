#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from ttkthemes import ThemedStyle
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
            value = float(prompt)
            return value
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid numeric value.")
            return None

def calculate_tubes():
    """
    Perform the Mould Tube Estimator calculations and display the result.
    """
    dia = get_float_input(dia_entry.get())
    charge = get_float_input(charge_entry.get())
    density = get_float_input(density_entry.get())

    if dia is not None and charge is not None and density is not None:
        predicted_tubes = predict_tubes(mlr_model, dia, charge, density)
        result_label.config(text="Number of Tubes Required: {:.2f}".format(predicted_tubes))

def main():
    """
    Main function to run the Mould Tube Estimator GUI.
    """

    # Load the dataset
    url = "https://raw.githubusercontent.com/Abhayekr/CW/main/Charge%20Weight10.csv"
    X, y = load_data(url)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model with hyperparameter optimization
    global mlr_model
    mlr_model = train_model(X_train, y_train)

    # Create the GUI window
    window = tk.Tk()
    window.title("Mould Tube Estimator")

    # Set the ttkthemes style
    style = ThemedStyle(window)
    style.set_theme("clearlooks")  # Choose the theme for the GUI (e.g., "arc", "equilux", "plastik")

    # Customize the style
    style.configure("TLabel", foreground="black", font=("Arial", 12))
    style.configure("TEntry", background="white", foreground="black", font=("Arial", 12))
    style.configure("TButton", background="darkblue", foreground="black", font=("Arial", 12))

    # Create the content frame
    content_frame = ttk.Frame(window, padding=20)
    content_frame.pack()

    # Create input labels and entry fields
    dia_label = ttk.Label(content_frame, text="Tube Diameter (in mm):")
    dia_label.grid(row=0, column=0, sticky=tk.W)

    global dia_entry
    dia_entry = ttk.Entry(content_frame)
    dia_entry.grid(row=0, column=1, pady=5)

    charge_label = ttk.Label(content_frame, text="Charge Weight (in kg):")
    charge_label.grid(row=1, column=0, sticky=tk.W)

    global charge_entry
    charge_entry = ttk.Entry(content_frame)
    charge_entry.grid(row=1, column=1, pady=5)

    density_label = ttk.Label(content_frame, text="JMAT Pro Density:")
    density_label.grid(row=2, column=0, sticky=tk.W)

    global density_entry
    density_entry = ttk.Entry(content_frame)
    density_entry.grid(row=2, column=1, pady=5)

    # Create the calculate button
    calculate_button = ttk.Button(content_frame, text="Calculate", command=calculate_tubes)
    calculate_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Create the result label
    global result_label
    result_label = ttk.Label(content_frame, text="")
    result_label.grid(row=4, column=0, columnspan=2, pady=5)

    # Run the GUI
    window.mainloop()

# Run the main function
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




