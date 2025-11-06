from pathlib import Path
import json
import joblib
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train(x_train, x_test, y_train, y_test, model_path):
    mlflow.set_experiment("MagnusIA_Experiment")
    with mlflow.start_run():
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / model_path
        pipeline_path = models_dir / "best_pipeline.joblib"
        label_map_path = models_dir / "label_index_to_class.json"

        pipeline = joblib.load(pipeline_path)
        with open(label_map_path, "r", encoding="utf-8") as f:
            idx_to_class = {int(k): v for k, v in json.load(f).items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}

        y_train_enc = y_train.map(class_to_idx)
        y_test_enc = y_test.map(class_to_idx)

        pipeline.fit(x_train, y_train_enc)

        y_pred_train_enc = pipeline.predict(x_train)
        y_pred_enc = pipeline.predict(x_test)
        acc_train = accuracy_score(y_train_enc, y_pred_train_enc)
        acc = accuracy_score(y_test_enc, y_pred_enc)
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_score(y_test_enc, y_pred_enc, average="macro"))
        mlflow.log_metric("precision_macro", precision_score(y_test_enc, y_pred_enc, average="macro"))
        mlflow.log_metric("recall_macro", recall_score(y_test_enc, y_pred_enc, average="macro"))

        mlflow.sklearn.log_model(pipeline, "model")