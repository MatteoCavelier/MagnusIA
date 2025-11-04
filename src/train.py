from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train(x_train, x_test, y_train, y_test):
    mlflow.set_experiment("MagnusIA_Experiment")
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 100
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        predictions = model.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))
        mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average="macro"))
        mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average="macro"))

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(model, "model")