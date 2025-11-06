"""
Simple Random Forest model training for comparison with XGBoost models.

This script trains a Random Forest classifier using the same data preprocessing
and evaluation pipeline as the XGBoost models, allowing for direct comparison.

Example usage:
    python -m src.train_rf --csv_path ./res/games.csv --experiment_name RF_Simple
"""

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
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

try:
    from .clean_data import clean_chess_data
except ImportError:
    from clean_data import clean_chess_data


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


def train_random_forest(
    csv_path: str = "./res/games.csv",
    experiment_name: str = "RF_Simple",
    test_size: float = 0.3,
    random_state: int = 42,
    use_duration: bool = True,
    use_victory_status: bool = True,
    use_turns: bool = True,
    moves_n: Optional[int] = None,
    moves_only_n: bool = True,
    moves_new_column: Optional[str] = None,
    moves_add_all_prefix: str = "moves_",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
):
    """
    Train a Random Forest classifier with the same preprocessing as XGBoost models.
    
    Args:
        csv_path: Path to the games CSV file
        experiment_name: MLflow experiment name
        test_size: Validation split proportion
        random_state: Random seed
        use_duration: Use duration dataset (True) or noduration (False)
        use_victory_status: Whether to keep the 'victory_status' column
        use_turns: Whether to keep the 'turns' column
        moves_n: Number of first moves to derive; None to disable
        moves_only_n: If True, keep exactly n moves; else create cumulative moves_1..n
        moves_new_column: Optional column name to store the truncated moves
        moves_add_all_prefix: Prefix for cumulative move columns when moves_only_n is False
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree (None for unlimited)
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        n_jobs: Number of jobs to run in parallel (-1 for all cores)
    
    Returns:
        Dictionary with model metrics and paths
    """
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

    # Label-encode targets
    le = LabelEncoder()
    train_y_enc = le.fit_transform(train_y)
    valid_y_enc = le.transform(valid_y)
    num_classes = len(le.classes_)

    # Build preprocessor based on feature columns
    preprocessor = build_preprocessor(X)

    # Derive a run/model dir name based on options (matches XGBoost format)
    subdir = f"{'duration' if use_duration else 'noduration'}-{'withturn' if use_turns else 'noturn'}-{'withvstatus' if use_victory_status else 'novstatus'}-{(('None') if moves_n is None else moves_n)}-{bool(moves_only_n)}"

    with mlflow.start_run(run_name=subdir):
        # Log dataset/options used for this run
        mlflow.log_params({
            "use_duration": use_duration,
            "use_victory_status": use_victory_status,
            "use_turns": use_turns,
            "moves_n": -1 if moves_n is None else moves_n,
            "moves_only_n": moves_only_n,
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "RandomForest",
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else "None",
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })

        # Create Random Forest classifier
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("rf", rf_clf),
        ])

        # Train the model
        pipeline.fit(train_x, train_y_enc)
        
        # Make predictions
        preds_train = pipeline.predict(train_x)
        preds_valid = pipeline.predict(valid_x)
        
        # Calculate metrics
        acc_train = accuracy_score(train_y_enc, preds_train)
        acc_valid = accuracy_score(valid_y_enc, preds_valid)
        f1 = f1_score(valid_y_enc, preds_valid, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": acc_train,
            "valid_accuracy": acc_valid,
            "best_accuracy": acc_valid,
            "best_train_accuracy": acc_train,
            "best_f1_macro": f1,
        })
        
        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Save label encoder classes to artifacts
        mlflow.log_dict({i: cls for i, cls in enumerate(le.classes_)}, artifact_file="label_index_to_class.json")

        # Persist model locally for later use (outside MLflow)
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / "models" / subdir
        models_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = models_dir / "best_pipeline.joblib"
        label_map_path = models_dir / "label_index_to_class.json"
        params_path = models_dir / "best_params.json"
        random_state_path = models_dir / "random_state.json"
        metrics_path = models_dir / "metrics.json"
    
        joblib.dump(pipeline, pipeline_path)
        # Also save in MLflow's local model format for portability
        mlflow.sklearn.save_model(pipeline, path=str(models_dir / "mlflow_model"))
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({int(i): str(cls) for i, cls in enumerate(le.classes_)}, f, ensure_ascii=False, indent=2)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }, f, ensure_ascii=False, indent=2)
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

        print(f"\n✓ Random Forest training completed!")
        print(f"  Train Accuracy: {acc_train:.4f}")
        print(f"  Valid Accuracy: {acc_valid:.4f}")
        print(f"  F1 Macro: {f1:.4f}")
        print(f"  Model saved to: {models_dir}")

        return {
            "train_accuracy": acc_train,
            "valid_accuracy": acc_valid,
            "f1_macro": f1,
            "local_model_dir": str(models_dir),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple Random Forest model for comparison with XGBoost.")
    parser.add_argument("--csv_path", type=str, default="./res/games.csv", help="Path to games CSV")
    parser.add_argument("--experiment_name", type=str, default="RF_Simple", help="MLflow experiment name")
    parser.add_argument("--test_size", type=float, default=0.3, help="Validation split proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--use_duration", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Use duration dataset (True) or noduration (False)")
    parser.add_argument("--use_victory_status", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Whether to keep the 'victory_status' column (True) or drop it (False)")
    parser.add_argument("--use_turns", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="Whether to keep the 'turns' column (True) or drop it (False)")
    parser.add_argument("--moves_n", type=int, default=None, help="Number of first moves to derive; None to disable")
    parser.add_argument("--moves_only_n", type=lambda v: str(v).lower() in ["1","true","yes","y"], default=True, help="If True, keep exactly n moves; else create cumulative moves_1..n")
    parser.add_argument("--moves_new_column", type=str, default=None, help="Optional column name to store the truncated moves")
    parser.add_argument("--moves_add_all_prefix", type=str, default="moves_", help="Prefix for cumulative move columns when moves_only_n is False")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree (None for unlimited)")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum number of samples required to split a node")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum number of samples required at a leaf node")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel (-1 for all cores)")

    args = parser.parse_args()

    train_random_forest(
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
        use_duration=args.use_duration,
        use_victory_status=args.use_victory_status,
        use_turns=args.use_turns,
        moves_n=args.moves_n,
        moves_only_n=args.moves_only_n,
        moves_new_column=args.moves_new_column,
        moves_add_all_prefix=args.moves_add_all_prefix,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
    )

