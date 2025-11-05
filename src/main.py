import mlflow
from flask import Flask, request, jsonify

app = Flask(__name__)

# 🔹 Charger le modèle pickle
print("model loading")
# 🔹 Charger le modèle pickle
model = mlflow.sklearn.load_model("../res/mlflow_model")
print("model loaded")


@app.route("/predict", methods=["POST"])
def predict():
    print("toto")
    data = request.get_json()
    preds = model.predict([data])
    return jsonify({"prediction": preds.tolist()})


if __name__ == "__main__":
    app.run(port=1234, debug=True)
