import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, render_template, request

def load_data(url):
    # Load the dataset from the given URL.
    data = pd.read_csv(url)
    X = data[["Tube Dia (mm)", "Charge Weight (kg)5", "JMAT Pro Density"]]
    y = data[["No. of Tubes6"]]
    return X, y

def train_model(X, y):
    pipe = make_pipeline(
        PolynomialFeatures(),
        LinearRegression()
    )

    param_grid = {'polynomialfeatures__degree': [2, 3, 4, 5]}
    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    return best_model

def predict_tubes(model, dia, charge, density):
    user_input = [[dia, charge, density]]
    predicted_tubes = float(model.predict(user_input))
    return predicted_tubes

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        dia = float(request.form["dia"])
        charge = float(request.form["charge"])
        density = float(request.form["density"])
        predicted_tubes = predict_tubes(mlr_model, dia, charge, density)
        return render_template("result.html", predicted_tubes=predicted_tubes)
    return render_template("index.html")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/Abhayekr/CW/main/Charge%20Weight10.csv"
    X, y = load_data(url)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    mlr_model = train_model(X_train, y_train)
    app.run(debug=True)
