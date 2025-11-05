from clean_data import get_train_split
from src.clean_data import clean_chess_data
from train import train
import sys
import subprocess
import mlflow

if __name__ == "__main__":
    # Entraîner le modèle
    print("Training model...")
    x_train, x_test, y_train, y_test = get_train_split(
        clean_chess_data(
            "./res/games.csv",
            moves_n=3,
            moves_only_n=False
        )["duration"]
    )
    train(x_train, x_test, y_train, y_test)

    # Récupérer le dernier run
    experiment = mlflow.get_experiment_by_name("MagnusIA_Experiment")
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )
    last_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{last_run_id}/model"

    print(f"\n{'='*60}")
    print(f"✓ Model trained successfully!")
    print(f"Run ID: {last_run_id}")
    print(f"{'='*60}\n")
    print(f"Starting MLflow server on http://127.0.0.1:1234")
    print("Press Ctrl+C to stop the server\n")

    # Utiliser le même Python qui exécute ce script
    try:
        subprocess.run(
            [sys.executable, "-m", "mlflow", "models", "serve",
             "-m", model_uri,
             "-p", "1234",
             "--no-conda"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
    except KeyboardInterrupt:
        print("\n\nServer stopped")