import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ active CORS pour toutes les routes

MODELS_DIR = r"../models"

LABELS = {0: "black", 1: "draw", 2: "white"}

COMMON_FIELDS = [
    "increment_code",
    "white_rating",
    "black_rating",
    "opening_eco",
    "opening_ply"
]

models = {}
models_info = {}

print("🔄 Chargement des modèles...")
for folder in os.listdir(MODELS_DIR):
    folder_path = os.path.join(MODELS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    try:
        model = mlflow.sklearn.load_model(os.path.join(folder_path, "mlflow_model"))
        models[folder] = model
        print(f"✅ Modèle chargé : {folder}")
    except Exception as e:
        print(f"❌ Erreur de chargement pour {folder}: {e}")

def expected_fields_from_model_name(model_name: str):
    parts = model_name.split("-")
    expected = list(COMMON_FIELDS)

    if parts[0] == "duration":
        expected.append("time")

    if len(parts) > 1 and parts[1] == "withturn":
        expected.append("turns")

    n_moves = None
    if len(parts) > 1 and parts[1].isdigit():
        n_moves = int(parts[1])
    elif len(parts) > 2 and parts[2].isdigit():
        n_moves = int(parts[2])

    if n_moves:
        for i in range(1, n_moves + 1):
            expected.append(f"moves_{i}")

    return expected

for model_name, model in models.items():
    route = f"/predict/{model_name}"
    expected = expected_fields_from_model_name(model_name)
    models_info[model_name] = {"expected_fields": expected}

    def make_predict(m, name, expected_fields):
        def predict():
            try:
                data = request.get_json()
                if not isinstance(data, dict):
                    return jsonify({"error": "JSON invalide ou non fourni."}), 400

                missing = [c for c in expected_fields if c not in data]
                if missing:
                    return jsonify({
                        "error": "Colonnes manquantes pour ce modèle.",
                        "model": name,
                        "missing_columns": missing,
                        "expected_columns": expected_fields
                    }), 400

                df = pd.DataFrame([data])
                preds = m.predict(df)
                mapped_preds = [LABELS.get(int(p), str(p)) for p in preds]

                return jsonify({
                    "model": name,
                    "prediction_index": preds.tolist(),
                    "prediction_label": mapped_preds
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return predict

    app.add_url_rule(route, route, make_predict(model, model_name, expected), methods=["POST"])
    print(f"➡ Route créée : {route} (attend {len(expected)} champs)")

@app.route("/models-info", methods=["GET"])
def info():
    return jsonify({k: v["expected_fields"] for k, v in models_info.items()})

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "API multi-modèles prête 🚀",
        "routes": [f"/predict/{k}" for k in models.keys()],
        "models_info_route": "/models-info"
    })

if __name__ == "__main__":
    app.run(port=1234, debug=True)
