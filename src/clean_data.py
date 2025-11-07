# Reusable cleaning functions based on the steps above
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

def get_train_split(dp):
    x = dp.drop(columns="winner")
    y = dp.get("winner")
    return train_test_split(x, y, random_state=42)


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """Load the CSV dataset from disk."""
    return pd.read_csv(file_path)


def filter_to_rated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rated games (rated == True)."""
    if 'rated' not in df.columns:
        return df.copy()
    return df[df['rated'] == True].copy()


def remove_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows sharing the same game `id` (keep first)."""
    if 'id' not in df.columns:
        return df.copy()
    return df.drop_duplicates(subset='id', keep='first').copy()


def remove_duplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates that share the same (created_at, white_id, black_id)."""
    required = {'created_at', 'white_id', 'black_id'}
    if not required.issubset(df.columns):
        return df.copy()
    return df.drop_duplicates(subset=['created_at', 'white_id', 'black_id'], keep='first').copy()


def add_game_duration_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Add `time` as seconds between `last_move_at` and `created_at`, then drop both timestamp columns."""
    cols = {'created_at', 'last_move_at'}
    out = df.copy()
    if cols.issubset(out.columns):
        out['time'] = (
                pd.to_datetime(out['last_move_at'], unit='ms') - pd.to_datetime(out['created_at'], unit='ms')
        ).dt.total_seconds()
        out = out.drop(columns=['last_move_at', 'created_at'])
    return out


def split_by_duration_variants(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create two variants:
    - dataset_duration: keep only rows where time != 0 and time != 10000.0
    - dataset_noduration: drop the `time` column entirely (keep all rows)
    If `time` is missing, returns (df copy, df copy without `time`).
    """
    out = df.copy()
    if 'time' in out.columns:
        dataset_duration = out[(out['time'] != 0) & (out['time'] != 10000.0)].copy()
        dataset_noduration = out.drop(columns=['time']).copy()
    else:
        dataset_duration = out.copy()
        dataset_noduration = out.copy()
    return dataset_duration, dataset_noduration


def drop_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop provided columns if present. No-op when `columns` is None or empty."""
    if not columns:
        return df.copy()
    return df.drop(columns=columns, errors='ignore').copy()


def first_k(s: str, fk: int) -> Optional[str]:
    if not isinstance(s, str):
        return None
    parts = s.split()
    if len(parts) == 0:
        return None
    return ' '.join(parts[:fk]) if len(parts) >= fk else s


def keep_first_n_moves(
        df: pd.DataFrame,
        n: int,
        only_n: bool = True,
        new_column: Optional[str] = None,
        add_all_prefix: str = 'moves_'
) -> pd.DataFrame:
    """Create truncated move sequences.

    - If only_n is True: keep exactly the first n SAN tokens. If `new_column` is None,
      overwrite `moves`; otherwise, write to `new_column`.
    - If only_n is False: add cumulative columns for k in [1..n] named
      f"{add_all_prefix}{k}", each containing the first k moves. The original
      `moves` column is not preserved.
    """
    if 'moves' not in df.columns:
        return df.copy()

    out = df.copy()

    if only_n:
        col = 'moves' if new_column is None else new_column
        out[col] = out['moves'].apply(lambda m: first_k(m, n))
    else:
        for k in range(1, max(1, n) + 1):
            out[f"{add_all_prefix}{k}"] = out['moves'].apply(lambda m: first_k(m, k))

    return out


def clean_chess_data(
        file_path: str,
        columns_to_drop: Optional[List[str]] = None,
        moves_n: Optional[int] = None,
        moves_only_n: bool = True,
        moves_new_column: Optional[str] = None,
        moves_add_all_prefix: str = 'moves_',
        drop_turns: bool = False,
        drop_victory_status: bool = False,
) -> Dict[str, pd.DataFrame]:
    """High-level pipeline that reproduces the notebook cleaning steps.

    Steps:
    1) load -> 2) filter rated -> 3) remove duplicate ids ->
    4) remove duplicate games by (created_at, white_id, black_id) ->
    5) add `time` (seconds) and drop raw timestamps ->
    6) create two variants: (duration != 0 AND != 10000.0) and (no `time`) ->
    7) optionally derive first-n moves (single or cumulative) ->
    8) drop requested columns in both variants.

    Returns a dict with keys: 'duration', 'noduration'.
    """
    df = load_raw_dataset(file_path)
    df = filter_to_rated(df)
    df = remove_duplicate_ids(df)
    df = remove_duplicate_games(df)
    df = add_game_duration_seconds(df)

    dataset_duration, dataset_noduration = split_by_duration_variants(df)

    # Optionally derive first-n moves on both variants before dropping columns
    if moves_n is not None and moves_n > 0:
        dataset_duration = keep_first_n_moves(
            dataset_duration,
            n=moves_n,
            only_n=moves_only_n,
            new_column=moves_new_column,
            add_all_prefix=moves_add_all_prefix,
        )
        dataset_noduration = keep_first_n_moves(
            dataset_noduration,
            n=moves_n,
            only_n=moves_only_n,
            new_column=moves_new_column,
            add_all_prefix=moves_add_all_prefix,
        )

    # Default columns to drop based on the notebook
    default_drop = ['id', 'white_id','black_id','opening_name', 'moves']
    # Optionally drop the number of turns column
    if drop_turns:
        default_drop.append('turns')
    # Optionally drop victory_status column
    if drop_victory_status:
        default_drop.append('victory_status')
    # Add rated to the columns to drop
    default_drop.append('rated')
    # Add columns to drop from the columns_to_drop list
    if columns_to_drop is not None:
        default_drop.extend(columns_to_drop)
    cols = columns_to_drop if columns_to_drop is not None else default_drop

    dataset_duration = drop_columns(dataset_duration, cols)
    dataset_noduration = drop_columns(dataset_noduration, cols)

    return {
        'duration': dataset_duration,
        'noduration': dataset_noduration,
    }