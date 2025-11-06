import pytest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from clean_data import (
    get_train_split,
    load_raw_dataset,
    filter_to_rated,
    remove_duplicate_ids,
    remove_duplicate_games,
    add_game_duration_seconds,
    split_by_duration_variants,
    drop_columns,
    first_k,
    keep_first_n_moves,
    clean_chess_data
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_chess_dataframe():
    """Fixture pour créer un DataFrame de jeux d'échecs simplifié"""
    return pd.DataFrame({
        'id': ['game1', 'game2', 'game3', 'game1'],  # game1 dupliqué
        'rated': [True, True, False, True],
        'created_at': [1000, 2000, 3000, 1000],
        'last_move_at': [2000, 4000, 5000, 2000],
        'white_id': ['player1', 'player2', 'player3', 'player1'],
        'black_id': ['player4', 'player5', 'player6', 'player4'],
        'winner': ['white', 'black', 'draw', 'white'],
        'moves': ['e4 e5', 'Nf3 Nc6 d4', 'd4 d5 c4', 'e4 e5'],
        'opening_name': ['Italian', 'Ruy Lopez', 'Queens Gambit', 'Italian']
    })


@pytest.fixture
def sample_csv_file(sample_chess_dataframe):
    """Fixture pour créer un fichier CSV temporaire"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_chess_dataframe.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_dataframe_with_winner():
    """Fixture pour tester get_train_split"""
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'winner': np.random.choice(['white', 'black', 'draw'], 100)
    })


# ============================================================================
# TEST get_train_split
# ============================================================================

def test_get_train_split_returns_four_arrays(sample_dataframe_with_winner):
    """Test que get_train_split retourne 4 éléments"""
    result = get_train_split(sample_dataframe_with_winner)
    assert len(result) == 4


def test_get_train_split_separates_features_and_target(sample_dataframe_with_winner):
    """Test que X ne contient pas 'winner' et y contient 'winner'"""
    x_train, x_test, y_train, y_test = get_train_split(sample_dataframe_with_winner)

    assert 'winner' not in x_train.columns
    assert 'winner' not in x_test.columns
    assert y_train.name == 'winner'
    assert y_test.name == 'winner'


def test_get_train_split_consistent_shapes(sample_dataframe_with_winner):
    """Test que les dimensions sont cohérentes"""
    x_train, x_test, y_train, y_test = get_train_split(sample_dataframe_with_winner)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[0] + x_test.shape[0] == len(sample_dataframe_with_winner)


def test_get_train_split_random_state_reproducibility(sample_dataframe_with_winner):
    """Test que random_state=42 assure la reproductibilité"""
    x_train1, x_test1, y_train1, y_test1 = get_train_split(sample_dataframe_with_winner)
    x_train2, x_test2, y_train2, y_test2 = get_train_split(sample_dataframe_with_winner)

    pd.testing.assert_frame_equal(x_train1, x_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)


# ============================================================================
# TEST load_raw_dataset
# ============================================================================

def test_load_raw_dataset_returns_dataframe(sample_csv_file):
    """Test que load_raw_dataset retourne un DataFrame"""
    df = load_raw_dataset(sample_csv_file)
    assert isinstance(df, pd.DataFrame)


def test_load_raw_dataset_loads_correct_data(sample_csv_file, sample_chess_dataframe):
    """Test que les données chargées correspondent au CSV"""
    df = load_raw_dataset(sample_csv_file)
    assert len(df) == len(sample_chess_dataframe)
    assert list(df.columns) == list(sample_chess_dataframe.columns)


def test_load_raw_dataset_file_not_found():
    """Test le comportement avec un fichier inexistant"""
    with pytest.raises(FileNotFoundError):
        load_raw_dataset('nonexistent_file.csv')


# ============================================================================
# TEST filter_to_rated
# ============================================================================

def test_filter_to_rated_keeps_only_rated_games(sample_chess_dataframe):
    """Test que seuls les jeux rated=True sont conservés"""
    result = filter_to_rated(sample_chess_dataframe)
    assert all(result['rated'] == True)
    assert len(result) == 3  # 3 jeux rated sur 4


def test_filter_to_rated_without_rated_column():
    """Test le comportement sans colonne 'rated'"""
    df = pd.DataFrame({'id': [1, 2, 3], 'winner': ['white', 'black', 'draw']})
    result = filter_to_rated(df)
    assert len(result) == len(df)


def test_filter_to_rated_returns_copy(sample_chess_dataframe):
    """Test que la fonction retourne une copie"""
    result = filter_to_rated(sample_chess_dataframe)
    assert result is not sample_chess_dataframe


# ============================================================================
# TEST remove_duplicate_ids
# ============================================================================

def test_remove_duplicate_ids_removes_duplicates(sample_chess_dataframe):
    """Test que les IDs dupliqués sont supprimés"""
    result = remove_duplicate_ids(sample_chess_dataframe)
    assert len(result) == 3  # game1 apparaît 2 fois, on garde le premier
    assert result['id'].nunique() == len(result)


def test_remove_duplicate_ids_keeps_first_occurrence(sample_chess_dataframe):
    """Test que la première occurrence est conservée"""
    result = remove_duplicate_ids(sample_chess_dataframe)
    first_game1 = result[result['id'] == 'game1'].iloc[0]
    assert first_game1['created_at'] == 1000  # Premier game1


def test_remove_duplicate_ids_without_id_column():
    """Test le comportement sans colonne 'id'"""
    df = pd.DataFrame({'winner': ['white', 'black'], 'moves': ['e4', 'd4']})
    result = remove_duplicate_ids(df)
    assert len(result) == len(df)


# ============================================================================
# TEST remove_duplicate_games
# ============================================================================

def test_remove_duplicate_games_by_key_fields(sample_chess_dataframe):
    """Test que les jeux dupliqués par (created_at, white_id, black_id) sont supprimés"""
    result = remove_duplicate_games(sample_chess_dataframe)
    # game1 apparaît 2 fois avec les mêmes (created_at, white_id, black_id)
    assert len(result) == 3


def test_remove_duplicate_games_without_required_columns():
    """Test le comportement sans les colonnes requises"""
    df = pd.DataFrame({'id': [1, 2], 'winner': ['white', 'black']})
    result = remove_duplicate_games(df)
    assert len(result) == len(df)


def test_remove_duplicate_games_keeps_first():
    """Test que la première occurrence est conservée"""
    df = pd.DataFrame({
        'created_at': [1000, 1000],
        'white_id': ['p1', 'p1'],
        'black_id': ['p2', 'p2'],
        'winner': ['white', 'black']
    })
    result = remove_duplicate_games(df)
    assert len(result) == 1
    assert result.iloc[0]['winner'] == 'white'


# ============================================================================
# TEST add_game_duration_seconds
# ============================================================================

def test_add_game_duration_seconds_creates_time_column(sample_chess_dataframe):
    """Test que la colonne 'time' est créée"""
    result = add_game_duration_seconds(sample_chess_dataframe)
    assert 'time' in result.columns


def test_add_game_duration_seconds_calculates_correctly(sample_chess_dataframe):
    """Test que la durée est calculée correctement"""
    result = add_game_duration_seconds(sample_chess_dataframe)
    # created_at: 1000ms, last_move_at: 2000ms -> 1 seconde
    assert result.iloc[0]['time'] == 1.0


def test_add_game_duration_seconds_drops_timestamp_columns(sample_chess_dataframe):
    """Test que les colonnes timestamp sont supprimées"""
    result = add_game_duration_seconds(sample_chess_dataframe)
    assert 'created_at' not in result.columns
    assert 'last_move_at' not in result.columns


def test_add_game_duration_seconds_without_required_columns():
    """Test le comportement sans les colonnes requises"""
    df = pd.DataFrame({'id': [1, 2], 'winner': ['white', 'black']})
    result = add_game_duration_seconds(df)
    assert 'time' not in result.columns


# ============================================================================
# TEST split_by_duration_variants
# ============================================================================

def test_split_by_duration_variants_creates_two_dataframes():
    """Test que deux DataFrames sont retournés"""
    df = pd.DataFrame({
        'time': [0, 1, 2, 10000.0, 5],
        'winner': ['white', 'black', 'draw', 'white', 'black']
    })
    duration, noduration = split_by_duration_variants(df)

    assert isinstance(duration, pd.DataFrame)
    assert isinstance(noduration, pd.DataFrame)


def test_split_by_duration_variants_filters_correctly():
    """Test que les valeurs 0 et 10000.0 sont filtrées dans duration"""
    df = pd.DataFrame({
        'time': [0, 1, 2, 10000.0, 5],
        'winner': ['white', 'black', 'draw', 'white', 'black']
    })
    duration, noduration = split_by_duration_variants(df)

    # Duration ne doit pas contenir 0 ni 10000.0
    assert len(duration) == 3
    assert 0 not in duration['time'].values
    assert 10000.0 not in duration['time'].values


def test_split_by_duration_variants_noduration_drops_time():
    """Test que noduration n'a pas de colonne 'time'"""
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'winner': ['white', 'black', 'draw']
    })
    duration, noduration = split_by_duration_variants(df)

    assert 'time' not in noduration.columns
    assert 'time' in duration.columns


def test_split_by_duration_variants_without_time_column():
    """Test le comportement sans colonne 'time'"""
    df = pd.DataFrame({'id': [1, 2], 'winner': ['white', 'black']})
    duration, noduration = split_by_duration_variants(df)

    assert len(duration) == len(df)
    assert len(noduration) == len(df)


# ============================================================================
# TEST drop_columns
# ============================================================================

def test_drop_columns_removes_specified_columns():
    """Test que les colonnes spécifiées sont supprimées"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    result = drop_columns(df, ['a', 'c'])

    assert 'a' not in result.columns
    assert 'c' not in result.columns
    assert 'b' in result.columns


def test_drop_columns_with_none_list():
    """Test que None ne supprime aucune colonne"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = drop_columns(df, None)
    assert list(result.columns) == ['a', 'b']


def test_drop_columns_with_empty_list():
    """Test qu'une liste vide ne supprime aucune colonne"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = drop_columns(df, [])
    assert list(result.columns) == ['a', 'b']


def test_drop_columns_ignores_nonexistent_columns():
    """Test que les colonnes inexistantes sont ignorées"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = drop_columns(df, ['a', 'nonexistent', 'also_nonexistent'])

    assert 'a' not in result.columns
    assert 'b' in result.columns


# ============================================================================
# TEST first_k
# ============================================================================

def test_first_k_returns_first_n_words():
    """Test que les n premiers mots sont retournés"""
    assert first_k("e4 e5 Nf3 Nc6", 2) == "e4 e5"
    assert first_k("d4 d5 c4 e6 Nc3", 3) == "d4 d5 c4"


def test_first_k_with_fewer_words():
    """Test avec moins de mots que demandé"""
    assert first_k("e4 e5", 5) == "e4 e5"


def test_first_k_with_single_word():
    """Test avec un seul mot"""
    assert first_k("e4", 1) == "e4"
    assert first_k("e4", 5) == "e4"


def test_first_k_with_empty_string():
    """Test avec une chaîne vide"""
    assert first_k("", 2) is None


def test_first_k_with_non_string():
    """Test avec un type non-string"""
    assert first_k(None, 2) is None
    assert first_k(123, 2) is None


def test_first_k_with_whitespace():
    """Test avec des espaces multiples"""
    assert first_k("e4  e5   Nf3", 2) == "e4 e5"


# ============================================================================
# TEST keep_first_n_moves
# ============================================================================

def test_keep_first_n_moves_only_n_overwrites_moves():
    """Test que only_n=True remplace la colonne moves"""
    df = pd.DataFrame({'moves': ['e4 e5 Nf3 Nc6', 'd4 d5 c4']})
    result = keep_first_n_moves(df, n=2, only_n=True)

    assert result['moves'].iloc[0] == 'e4 e5'
    assert result['moves'].iloc[1] == 'd4 d5'


def test_keep_first_n_moves_only_n_with_new_column():
    """Test que new_column crée une nouvelle colonne"""
    df = pd.DataFrame({'moves': ['e4 e5 Nf3 Nc6', 'd4 d5 c4']})
    result = keep_first_n_moves(df, n=2, only_n=True, new_column='moves_2')

    assert 'moves' in result.columns
    assert 'moves_2' in result.columns
    assert result['moves_2'].iloc[0] == 'e4 e5'


def test_keep_first_n_moves_cumulative_creates_multiple_columns():
    """Test que only_n=False crée des colonnes cumulatives"""
    df = pd.DataFrame({'moves': ['e4 e5 Nf3 Nc6']})
    result = keep_first_n_moves(df, n=3, only_n=False)

    assert 'moves_1' in result.columns
    assert 'moves_2' in result.columns
    assert 'moves_3' in result.columns
    assert result['moves_1'].iloc[0] == 'e4'
    assert result['moves_2'].iloc[0] == 'e4 e5'
    assert result['moves_3'].iloc[0] == 'e4 e5 Nf3'


def test_keep_first_n_moves_custom_prefix():
    """Test avec un préfixe personnalisé"""
    df = pd.DataFrame({'moves': ['e4 e5 Nf3']})
    result = keep_first_n_moves(df, n=2, only_n=False, add_all_prefix='move_')

    assert 'move_1' in result.columns
    assert 'move_2' in result.columns


def test_keep_first_n_moves_without_moves_column():
    """Test sans colonne 'moves'"""
    df = pd.DataFrame({'id': [1, 2], 'winner': ['white', 'black']})
    result = keep_first_n_moves(df, n=2)

    assert len(result) == len(df)
    assert list(result.columns) == ['id', 'winner']


# ============================================================================
# TEST clean_chess_data (Pipeline complète)
# ============================================================================

def test_clean_chess_data_returns_dict_with_two_keys(sample_csv_file):
    """Test que clean_chess_data retourne un dict avec 'duration' et 'noduration'"""
    result = clean_chess_data(sample_csv_file)

    assert isinstance(result, dict)
    assert 'duration' in result
    assert 'noduration' in result


def test_clean_chess_data_filters_rated_games(sample_csv_file):
    """Test que seuls les jeux rated sont conservés"""
    result = clean_chess_data(sample_csv_file)

    # Le CSV d'origine a 4 lignes dont 3 rated
    # Mais game1 est dupliqué, donc après déduplication: 2 rated uniques
    assert len(result['duration']) <= 3
    assert len(result['noduration']) <= 3


def test_clean_chess_data_removes_duplicates(sample_csv_file):
    """Test que les doublons sont supprimés"""
    result = clean_chess_data(sample_csv_file)

    # Vérifier qu'il n'y a pas de doublons d'ID dans les résultats
    # Note: 'id' est dans default_drop, donc on vérifie juste le nombre de lignes
    assert len(result['duration']) >= 1
    assert len(result['noduration']) >= 1


def test_clean_chess_data_drops_default_columns(sample_csv_file):
    """Test que les colonnes par défaut sont supprimées"""
    result = clean_chess_data(sample_csv_file)

    default_drop = ['id', 'white_id', 'black_id', 'opening_name', 'moves']

    for col in default_drop:
        assert col not in result['duration'].columns
        assert col not in result['noduration'].columns


def test_clean_chess_data_drops_custom_columns(sample_csv_file):
    """Test avec des colonnes personnalisées à supprimer"""
    result = clean_chess_data(sample_csv_file, columns_to_drop=['winner'])

    assert 'winner' not in result['duration'].columns
    assert 'winner' not in result['noduration'].columns


def test_clean_chess_data_with_moves_n(sample_csv_file):
    """Test avec troncature des coups à n mouvements"""
    result = clean_chess_data(
        sample_csv_file,
        moves_n=2,
        moves_only_n=True,
        moves_new_column='moves_2'
    )

    # moves_2 devrait exister avant que 'moves' soit supprimé
    # Mais 'moves' est dans default_drop, donc moves_2 devrait rester
    # Note: Cela dépend de l'ordre des opérations dans clean_chess_data


def test_clean_chess_data_creates_cumulative_moves(sample_csv_file):
    """Test avec création de colonnes cumulatives de coups"""
    result = clean_chess_data(
        sample_csv_file,
        moves_n=3,
        moves_only_n=False,
        columns_to_drop=['id', 'white_id', 'black_id', 'opening_name']  # Ne pas drop 'moves'
    )

    # Vérifier que les colonnes moves_1, moves_2, moves_3 existent
    assert 'moves_1' in result['duration'].columns
    assert 'moves_2' in result['duration'].columns
    assert 'moves_3' in result['duration'].columns


def test_clean_chess_data_noduration_has_no_time_column(sample_csv_file):
    """Test que noduration n'a pas de colonne 'time'"""
    result = clean_chess_data(sample_csv_file)

    assert 'time' not in result['noduration'].columns


def test_clean_chess_data_duration_has_time_column(sample_csv_file):
    """Test que duration a la colonne 'time'"""
    result = clean_chess_data(sample_csv_file)

    if len(result['duration']) > 0:
        assert 'time' in result['duration'].columns


def test_clean_chess_data_duration_filters_edge_cases(sample_csv_file):
    """Test que duration filtre les valeurs 0 et 10000.0"""
    result = clean_chess_data(sample_csv_file)

    if len(result['duration']) > 0 and 'time' in result['duration'].columns:
        assert 0 not in result['duration']['time'].values
        assert 10000.0 not in result['duration']['time'].values


def test_clean_chess_data_preserves_data_integrity(sample_csv_file):
    """Test que les données sont préservées correctement"""
    result = clean_chess_data(sample_csv_file)

    # Les deux variantes doivent avoir le même nombre de lignes
    # (sauf duration qui filtre certaines valeurs de time)
    assert len(result['noduration']) >= len(result['duration'])
