from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
        prediction = model.predict([features])[0]
        species = ["setosa", "versicolor", "virginica"]
        result = species[prediction]
        return render_template("index.html", prediction_text=f"The predicted species is: {result}")
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
