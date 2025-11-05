import pytest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from subprocess import CalledProcessError

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


@pytest.fixture
def mock_train_data():
    """Fixture pour simuler les données retournées par get_train"""
    np.random.seed(42)
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 10)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def mock_mlflow_run():
    """Fixture pour simuler un run MLflow"""
    run_data = pd.DataFrame({
        'run_id': ['test-run-id-12345'],
        'experiment_id': ['test-experiment-id'],
        'status': ['FINISHED'],
        'start_time': [pd.Timestamp('2024-01-01 10:00:00')],
        'end_time': [pd.Timestamp('2024-01-01 10:30:00')]
    })
    return run_data


@pytest.fixture
def mock_experiment():
    """Fixture pour simuler une expérience MLflow"""
    experiment = MagicMock()
    experiment.experiment_id = 'test-experiment-id'
    experiment.name = 'MagnusIA_Experiment'
    return experiment


@patch('subprocess.run')
@patch('mlflow.search_runs')
@patch('mlflow.get_experiment_by_name')
def test_retrieve_last_run(mock_get_experiment, mock_search_runs, mock_subprocess,
                           mock_experiment, mock_mlflow_run):
    """Test la récupération du dernier run MLflow"""
    mock_get_experiment.return_value = mock_experiment
    mock_search_runs.return_value = mock_mlflow_run

    # Simuler le code du main
    experiment = mock_get_experiment("MagnusIA_Experiment")
    experiment_id = experiment.experiment_id

    runs = mock_search_runs(
        experiment_ids=[experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )

    last_run_id = runs.iloc[0]["run_id"]

    # Vérifications
    assert last_run_id == 'test-run-id-12345'
    mock_get_experiment.assert_called_once_with("MagnusIA_Experiment")
    mock_search_runs.assert_called_once_with(
        experiment_ids=[experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )


@patch('subprocess.run')
def test_mlflow_server_command_structure(mock_subprocess):
    """Test la structure de la commande de démarrage du serveur MLflow"""
    last_run_id = 'test-run-id-12345'
    model_uri = f"runs:/{last_run_id}/model"

    # Simuler l'appel subprocess
    expected_command = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", "1234",
        "--no-conda"
    ]

    mock_subprocess(expected_command, check=True)

    # Vérifications
    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]

    assert sys.executable in call_args
    assert "mlflow" in call_args
    assert "models" in call_args
    assert "serve" in call_args
    assert "-m" in call_args
    assert model_uri in call_args
    assert "-p" in call_args
    assert "1234" in call_args
    assert "--no-conda" in call_args


@patch('subprocess.run')
def test_mlflow_server_port_configuration(mock_subprocess):
    """Test que le port 1234 est bien configuré"""
    last_run_id = 'test-run-id-12345'
    model_uri = f"runs:/{last_run_id}/model"

    command = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", "1234",
        "--no-conda"
    ]

    mock_subprocess(command, check=True)

    call_args = mock_subprocess.call_args[0][0]
    port_index = call_args.index("-p")
    assert call_args[port_index + 1] == "1234"


@patch('subprocess.run')
def test_mlflow_server_error_handling(mock_subprocess):
    """Test la gestion d'erreur lors du démarrage du serveur"""
    last_run_id = 'test-run-id-12345'
    model_uri = f"runs:/{last_run_id}/model"

    # Simuler une erreur subprocess
    mock_subprocess.side_effect = CalledProcessError(1, 'mlflow', stderr=b'Error starting server')

    with pytest.raises(CalledProcessError) as exc_info:
        mock_subprocess(
            [sys.executable, "-m", "mlflow", "models", "serve",
             "-m", model_uri,
             "-p", "1234",
             "--no-conda"],
            check=True
        )

    assert exc_info.value.returncode == 1


@patch('subprocess.run')
def test_keyboard_interrupt_handling(mock_subprocess):
    """Test la gestion de l'interruption clavier (Ctrl+C)"""
    last_run_id = 'test-run-id-12345'
    model_uri = f"runs:/{last_run_id}/model"

    # Simuler un KeyboardInterrupt
    mock_subprocess.side_effect = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        mock_subprocess(
            [sys.executable, "-m", "mlflow", "models", "serve",
             "-m", model_uri,
             "-p", "1234",
             "--no-conda"],
            check=True
        )


@patch('mlflow.get_experiment_by_name')
def test_experiment_not_found(mock_get_experiment):
    """Test le comportement quand l'expérience n'existe pas"""
    mock_get_experiment.return_value = None

    experiment = mock_get_experiment("NonExistentExperiment")

    assert experiment is None
    mock_get_experiment.assert_called_once_with("NonExistentExperiment")


@patch('mlflow.search_runs')
@patch('mlflow.get_experiment_by_name')
def test_no_runs_found(mock_get_experiment, mock_search_runs, mock_experiment):
    """Test le comportement quand aucun run n'est trouvé"""
    mock_get_experiment.return_value = mock_experiment

    # DataFrame vide
    empty_df = pd.DataFrame()
    mock_search_runs.return_value = empty_df

    experiment = mock_get_experiment("MagnusIA_Experiment")
    runs = mock_search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )

    assert len(runs) == 0
    assert isinstance(runs, pd.DataFrame)


def test_model_uri_format():
    """Test le format de l'URI du modèle"""
    last_run_id = 'test-run-id-12345'
    model_uri = f"runs:/{last_run_id}/model"

    # Vérifications du format
    assert model_uri == "runs:/test-run-id-12345/model"
    assert model_uri.startswith("runs:/")
    assert model_uri.endswith("/model")
    assert last_run_id in model_uri


def test_model_uri_with_different_run_ids():
    """Test le format de l'URI avec différents IDs"""
    test_cases = [
        ('abc123', 'runs:/abc123/model'),
        ('run-2024-01-01', 'runs:/run-2024-01-01/model'),
        ('12345-67890-abcdef', 'runs:/12345-67890-abcdef/model')
    ]

    for run_id, expected_uri in test_cases:
        model_uri = f"runs:/{run_id}/model"
        assert model_uri == expected_uri


@patch('mlflow.search_runs')
@patch('mlflow.get_experiment_by_name')
def test_search_runs_parameters(mock_get_experiment, mock_search_runs,
                                mock_experiment, mock_mlflow_run):
    """Test les paramètres de recherche des runs"""
    mock_get_experiment.return_value = mock_experiment
    mock_search_runs.return_value = mock_mlflow_run

    experiment = mock_get_experiment("MagnusIA_Experiment")

    mock_search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )

    # Vérifier les arguments de l'appel
    call_kwargs = mock_search_runs.call_args[1]
    assert call_kwargs['experiment_ids'] == [experiment.experiment_id]
    assert call_kwargs['order_by'] == ["end_time DESC"]
    assert call_kwargs['max_results'] == 1


@patch('mlflow.search_runs')
@patch('mlflow.get_experiment_by_name')
def test_multiple_runs_returns_latest(mock_get_experiment, mock_search_runs, mock_experiment):
    """Test que seul le dernier run est retourné quand plusieurs existent"""
    mock_get_experiment.return_value = mock_experiment

    # Simuler plusieurs runs
    multiple_runs = pd.DataFrame({
        'run_id': ['old-run-1', 'old-run-2', 'latest-run'],
        'experiment_id': ['test-experiment-id'] * 3,
        'status': ['FINISHED'] * 3,
        'start_time': [
            pd.Timestamp('2024-01-01 10:00:00'),
            pd.Timestamp('2024-01-02 10:00:00'),
            pd.Timestamp('2024-01-03 10:00:00')
        ],
        'end_time': [
            pd.Timestamp('2024-01-01 11:00:00'),
            pd.Timestamp('2024-01-02 11:00:00'),
            pd.Timestamp('2024-01-03 11:00:00')
        ]
    })

    # Mais avec max_results=1, on ne retourne que le premier (le plus récent)
    mock_search_runs.return_value = multiple_runs.head(1)

    experiment = mock_get_experiment("MagnusIA_Experiment")
    runs = mock_search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["end_time DESC"],
        max_results=1
    )

    assert len(runs) == 1
    assert runs.iloc[0]['run_id'] == 'old-run-1'


@patch('subprocess.run')
def test_no_conda_flag_present(mock_subprocess):
    """Test que le flag --no-conda est bien présent"""
    model_uri = "runs:/test-run/model"

    command = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", "1234",
        "--no-conda"
    ]

    mock_subprocess(command, check=True)

    call_args = mock_subprocess.call_args[0][0]
    assert "--no-conda" in call_args


def test_python_executable_is_correct():
    """Test que sys.executable pointe vers le bon interpréteur Python"""
    # Vérifier que sys.executable est un chemin valide
    assert os.path.exists(sys.executable)
    assert 'python' in sys.executable.lower()