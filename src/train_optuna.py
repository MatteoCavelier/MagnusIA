import optuna
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path
import json
import joblib
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from .clean_data import clean_chess_data


def get_or_create_experiment(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", "passthrough", num_cols),
    ])


def load_features_target(
    csv_path: str,
    *,
    use_duration: bool = True,
    use_victory_status: bool = True,
    use_turns: bool = True,
    moves_n: Optional[int] = None,
    moves_only_n: bool = True,
    moves_new_column: Optional[str] = None,
    moves_add_all_prefix: str = "moves_",
) -> Tuple[pd.DataFrame, pd.Series]:
    datasets = clean_chess_data(
        csv_path,
        moves_n=moves_n,
        moves_only_n=moves_only_n,
        moves_new_column=moves_new_column,
        moves_add_all_prefix=moves_add_all_prefix,
        drop_turns=not use_turns,
        drop_victory_status=not use_victory_status,
    )
    key = "duration" if use_duration else "noduration"
    df = datasets[key].drop(columns=["rated"], errors="ignore").copy()
    assert "winner" in df.columns, f"winner not in: {df.columns.tolist()}"
    X = df.drop(columns=["winner"]).copy()
    y = df["winner"].copy()
    return X, y


def objective_factory(train_x: pd.DataFrame, valid_x: pd.DataFrame, train_y_enc, valid_y_enc, preprocessor: ColumnTransformer, num_classes: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            tree_method="hist",
            n_jobs=0,
            **params,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", clf),
        ])

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            pipeline.fit(train_x, train_y_enc)
            preds_train = pipeline.predict(train_x)
            preds_valid = pipeline.predict(valid_x)
            acc_train = accuracy_score(train_y_enc, preds_train)
            acc_valid = accuracy_score(valid_y_enc, preds_valid)
            f1_valid = f1_score(valid_y_enc, preds_valid, average="macro")
            mlflow.log_metrics({
                "train_accuracy": acc_train,
                "valid_accuracy": acc_valid,
                "f1_macro": f1_valid,
            })
        # Optimize for f1 by default
        return f1_valid

    return objective


def run_study(
    csv_path: str = "./res/games.csv",
    experiment_name: str = "XGB_Optuna_MLflow",
    n_trials: int = 20,
    test_size: float = 0.3,
    random_state: int = 42,
    use_duration: bool = True,
    use_victory_status: bool = True,
    use_turns: bool = True,
    moves_n: Optional[int] = None,
    moves_only_n: bool = True,
    moves_new_column: Optional[str] = None,
    moves_add_all_prefix: str = "moves_",
):
    mlflow.set_experiment(experiment_name)
    X, y = load_features_target(
        csv_path,
        use_duration=use_duration,
        use_victory_status=use_victory_status,
        use_turns=use_turns,
        moves_n=moves_n,
        moves_only_n=moves_only_n,
        moves_new_column=moves_new_column,
        moves_add_all_prefix=moves_add_all_prefix,
    )

    train_x, valid_x, train_y, valid_y = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    # Label-encode targets once for all trials
    le = LabelEncoder()
    train_y_enc = le.fit_transform(train_y)
    valid_y_enc = le.transform(valid_y)
    num_classes = len(le.classes_)

    # Build preprocessor based on feature columns (no target)
    preprocessor = build_preprocessor(X)

    # Derive a run/model dir name based on options (matches local save dir)
    # Include withturn/noturn to indicate whether 'turns' column is used
    subdir = f"{'duration' if use_duration else 'noduration'}-{'withturn' if use_turns else 'noturn'}-{'withvstatus' if use_victory_status else 'novstatus'}-{(('None') if moves_n is None else moves_n)}-{bool(moves_only_n)}"

    with mlflow.start_run(run_name=subdir):
        # Log dataset/options used for this study
        mlflow.log_params({
            "use_duration": use_duration,
            "use_victory_status": use_victory_status,
            "use_turns": use_turns,
            "moves_n": -1 if moves_n is None else moves_n,
            "moves_only_n": moves_only_n,
            "test_size": test_size,
            "random_state": random_state,
        })
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective_factory(train_x, valid_x, train_y_enc, valid_y_enc, preprocessor, num_classes),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # Train best model on the same split and log it
        best_params = study.best_trial.params
        best_clf = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            tree_method="hist",
            n_jobs=0,
            **best_params,
        )
        best_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", best_clf),
        ])
        best_pipeline.fit(train_x, train_y_enc)
        preds_train = best_pipeline.predict(train_x)
        preds_valid = best_pipeline.predict(valid_x)
        acc_train = accuracy_score(train_y_enc, preds_train)
        acc_valid = accuracy_score(valid_y_enc, preds_valid)
        f1 = f1_score(valid_y_enc, preds_valid, average="macro")

        mlflow.log_params(best_params)
        # Log the random_state as well
        mlflow.log_param("random_state", random_state)
        mlflow.log_metrics({
            "best_train_accuracy": acc_train,
            "best_accuracy": acc_valid,
            "best_f1_macro": f1,
        })
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        # Save label encoder classes to artifacts
        mlflow.log_dict({i: cls for i, cls in enumerate(le.classes_)}, artifact_file="label_index_to_class.json")

        # Persist best model locally for later use (outside MLflow)
        # Store under an options-specific subfolder: use_duration, moves_n, moves_only_n (in that order)
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / "models" / subdir
        models_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = models_dir / "best_pipeline.joblib"
        label_map_path = models_dir / "label_index_to_class.json"
        params_path = models_dir / "best_params.json"
        random_state_path = models_dir / "random_state.json"
        metrics_path = models_dir / "metrics.json"
    
        joblib.dump(best_pipeline, pipeline_path)
        # Also save in MLflow's local model format for portability
        mlflow.sklearn.save_model(best_pipeline, path=str(models_dir / "mlflow_model"))
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({int(i): str(cls) for i, cls in enumerate(le.classes_)}, f, ensure_ascii=False, indent=2)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        with open(random_state_path, "w", encoding="utf-8") as f:
            json.dump(random_state, f, ensure_ascii=False, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_accuracy": acc_valid,
                "best_f1_macro": f1,
                "best_train_accuracy": acc_train,
            }, f, ensure_ascii=False, indent=2)

        # Log where we saved locally for convenience
        mlflow.log_param("local_model_dir", str(models_dir))

        # Attach best metrics to the returned study for downstream use
        try:
            study.best_metrics = {
                "best_accuracy": acc_valid,
                "best_f1_macro": f1,
                "best_train_accuracy": acc_train,
                "local_model_dir": str(models_dir),
            }
        except Exception:
            pass

        return study

# Example usage: python -m src.train_optuna --n_trials 10 --use_duration true --moves_n 5 --moves_only_n false
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna study for XGBoost with MLflow logging.")
    parser.add_argument("--csv_path", type=str, default="./res/games.csv", help="Path to games CSV")
    parser.add_argument("--experiment_name", type=str, default="XGB_Optuna_MLflow", help="MLflow experiment name")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--test_size", type=float, default=0.3, help="Validation split proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--use_duration", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Use duration dataset (True) or noduration (False)")
    parser.add_argument("--use_victory_status", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Whether to keep the 'victory_status' column (True) or drop it (False)")
    parser.add_argument("--use_turns", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Whether to keep the 'turns' column (True) or drop it (False)")
    parser.add_argument("--moves_n", type=int, default=None, help="Number of first moves to derive; None to disable")
    parser.add_argument("--moves_only_n", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="If True, keep exactly n moves; else create cumulative moves_1..n")
    parser.add_argument("--moves_new_column", type=str, default=None, help="Optional column name to store the truncated moves")
    parser.add_argument("--moves_add_all_prefix", type=str, default="moves_", help="Prefix for cumulative move columns when moves_only_n is False")

    args = parser.parse_args()

    run_study(
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        test_size=args.test_size,
        random_state=args.random_state,
        use_duration=args.use_duration,
        use_victory_status=args.use_victory_status,
        use_turns=args.use_turns,
        moves_n=args.moves_n,
        moves_only_n=args.moves_only_n,
        moves_new_column=args.moves_new_column,
        moves_add_all_prefix=args.moves_add_all_prefix,
    )