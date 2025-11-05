import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier # Ajout pour le test de type

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# L'importation suppose que le fichier source est 'src/train_ai.py'
from train_ai import train

@pytest.fixture
def sample_data():
    """Fixture pour créer des données d'entraînement et de test fictives"""
    np.random.seed(42)
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 10)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)
    return x_train, x_test, y_train, y_test


# --- Décorateurs Patch Corrigés ---

@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_logs_params(mock_log_model, mock_log_metric, mock_log_param,
                           mock_set_experiment, mock_start_run, sample_data):
    """Test que les paramètres sont bien loggés dans MLflow"""
    x_train, x_test, y_train, y_test = sample_data

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    train(x_train, x_test, y_train, y_test)

    # Vérifier que set_experiment a été appelé
    mock_set_experiment.assert_called_once_with("MagnusIA_Experiment")

    # Vérifier que les paramètres ont été loggés
    assert mock_log_param.call_count == 2
    mock_log_param.assert_any_call("n_estimators", 100)
    mock_log_param.assert_any_call("max_depth", 100)


@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_logs_metrics(mock_log_model, mock_log_metric, mock_log_param,
                            mock_set_experiment, mock_start_run, sample_data):
    """Test que les métriques sont bien loggées dans MLflow"""
    x_train, x_test, y_train, y_test = sample_data

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    train(x_train, x_test, y_train, y_test)

    # Vérifier que 4 métriques ont été loggées
    assert mock_log_metric.call_count == 4

    # Récupérer les noms des métriques loggées
    logged_metrics = [call[0][0] for call in mock_log_metric.call_args_list]
    assert "accuracy" in logged_metrics
    assert "f1_macro" in logged_metrics
    assert "precision_macro" in logged_metrics
    assert "recall_macro" in logged_metrics

    # Vérifier que toutes les valeurs de métriques sont entre 0 et 1
    for call in mock_log_metric.call_args_list:
        metric_value = call[0][1]
        assert 0 <= metric_value <= 1, f"Metric value {metric_value} is out of range [0, 1]"


@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_logs_model(mock_log_model, mock_log_metric, mock_log_param,
                          mock_set_experiment, mock_start_run, sample_data):
    """Test que le modèle est bien sauvegardé"""
    x_train, x_test, y_train, y_test = sample_data

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    train(x_train, x_test, y_train, y_test)

    # Vérifier que le modèle a été loggé avec le bon nom
    mock_log_model.assert_called_once()
    args = mock_log_model.call_args[0]
    assert args[1] == "model"

    # Vérifier que le premier argument est bien un modèle RandomForest
    assert isinstance(args[0], RandomForestClassifier)


@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_model_predictions(mock_log_model, mock_log_metric, mock_log_param,
                                 mock_set_experiment, mock_start_run, sample_data):
    """Test que le modèle fait des prédictions valides"""
    x_train, x_test, y_train, y_test = sample_data

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    train(x_train, x_test, y_train, y_test)

    # Récupérer le modèle sauvegardé
    model = mock_log_model.call_args[0][0]

    # Faire des prédictions
    predictions = model.predict(x_test)

    # Vérifier que les prédictions ont la bonne forme
    assert predictions.shape[0] == y_test.shape[0]

    # Vérifier que les prédictions sont binaires (0 ou 1)
    assert set(predictions).issubset({0, 1})


@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_random_state_consistency(mock_log_model, mock_log_metric, mock_log_param,
                                        mock_set_experiment, mock_start_run, sample_data):
    """Test que le random_state=42 assure la reproductibilité"""
    x_train, x_test, y_train, y_test = sample_data

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    # Premier entraînement
    train(x_train, x_test, y_train, y_test)
    model1 = mock_log_model.call_args[0][0]
    predictions1 = model1.predict(x_test)

    # Reset des mocks
    mock_log_model.reset_mock()

    # Deuxième entraînement
    train(x_train, x_test, y_train, y_test)
    model2 = mock_log_model.call_args[0][0]
    predictions2 = model2.predict(x_test)

    # Vérifier que les prédictions sont identiques
    assert np.array_equal(predictions1, predictions2), "Random state not working, predictions differ"


def test_train_with_empty_data():
    """Test le comportement avec des données vides"""
    x_train = np.array([]).reshape(0, 10)
    x_test = np.array([]).reshape(0, 10)
    y_train = np.array([])
    y_test = np.array([])

    with pytest.raises(ValueError):
        train(x_train, x_test, y_train, y_test)


def test_train_with_mismatched_dimensions():
    """Test le comportement avec des dimensions incompatibles"""
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 5)  # Nombre de features différent
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)

    with pytest.raises(ValueError):
        train(x_train, x_test, y_train, y_test)


def test_train_with_mismatched_samples():
    """Test le comportement avec un nombre d'échantillons incompatible"""
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 10)
    y_train = np.random.randint(0, 2, 50)  # Nombre différent
    y_test = np.random.randint(0, 2, 20)

    with pytest.raises(ValueError):
        train(x_train, x_test, y_train, y_test)


@patch('train_ai.mlflow.start_run')
@patch('train_ai.mlflow.set_experiment')
@patch('train_ai.mlflow.log_param')
@patch('train_ai.mlflow.log_metric')
@patch('train_ai.mlflow.sklearn.log_model')
def test_train_with_multiclass_data(mock_log_model, mock_log_metric, mock_log_param,
                                    mock_set_experiment, mock_start_run):
    """Test avec des données multi-classes (3 classes)"""
    np.random.seed(42)
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 10)
    y_train = np.random.randint(0, 3, 100)  # 3 classes
    y_test = np.random.randint(0, 3, 20)

    # Mock du context manager
    mock_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_context
    mock_start_run.return_value.__exit__.return_value = None

    train(x_train, x_test, y_train, y_test)

    # Vérifier que l'entraînement s'est bien passé
    mock_set_experiment.assert_called_once()
    assert mock_log_metric.call_count == 4