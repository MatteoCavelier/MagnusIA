import mlflow
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 🔹 Charger le modèle pickle
print("model loading")
# 🔹 Charger le modèle pickle
model = mlflow.sklearn.load_model("C:\School\MagnusIA\models\duration-5-False\mlflow_model")
print("model loaded")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Convert JSON to pandas DataFrame with one row
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return jsonify({"prediction": preds.tolist()})


if __name__ == "__main__":
    app.run(port=1234)
