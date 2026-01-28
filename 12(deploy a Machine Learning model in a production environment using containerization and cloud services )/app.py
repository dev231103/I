from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# HTML embedded directly in Python
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction App</title>
</head>
<body style="font-family: Arial; padding: 40px;">
    <h1>Iris Flower Class Predictor</h1>

    <form method="POST">
        <label>Sepal Length:</label><br>
        <input type="text" name="f1" required><br><br>

        <label>Sepal Width:</label><br>
        <input type="text" name="f2" required><br><br>

        <label>Petal Length:</label><br>
        <input type="text" name="f3" required><br><br>

        <label>Petal Width:</label><br>
        <input type="text" name="f4" required><br><br>

        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
        <h2>{{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    # Map class index to name
    class_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    if request.method == "POST":
        try:
            f1 = float(request.form["f1"])
            f2 = float(request.form["f2"])
            f3 = float(request.form["f3"])
            f4 = float(request.form["f4"])

            data = np.array([[f1, f2, f3, f4]])
            pred = model.predict(data)[0]

            prediction = f"The flower belongs to: {class_names[pred]}"

        except:
            prediction = "Invalid input. Please enter numeric values."

    return render_template_string(HTML_PAGE, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
