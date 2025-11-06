"""
Run Optuna hyperparameter optimization for multiple dataset configurations.

This script runs Optuna studies for different combinations of:
- With duration / without duration
- Different numbers of moves (e.g., 3, 5, 10)

By default, the script will run studies for all combinations of the following:
- With duration / without duration
- With turns / without turns
- Different numbers of moves ([None, 1, 2, 3, 4, 5])

Example usage:
    python -m src.train_optuna_multi --n_trials 10 --moves_n 1 2 3 4 5
"""

import argparse
from typing import List, Optional, Tuple
from pathlib import Path
import json

from .train_optuna import run_study


def run_multi_study(
    csv_path: str = "./res/games.csv",
    experiment_name: str = "XGB_Optuna_Multi_Config",
    n_trials: int = 20,
    test_size: float = 0.3,
    random_state: int = 42,
    use_duration_list: Optional[List[bool]] = None,
    use_turns_list: Optional[List[bool]] = None,
    use_victory_status_list: Optional[List[bool]] = None,
    moves_n_list: Optional[List[Optional[int]]] = None,
    moves_only_n: bool = True,
    moves_new_column: Optional[str] = None,
    moves_add_all_prefix: str = "moves_",
):
    """
    Run Optuna studies for multiple dataset configurations.
    
    Args:
        csv_path: Path to the games CSV file
        experiment_name: Base MLflow experiment name (each config gets a sub-experiment)
        n_trials: Number of Optuna trials per configuration
        test_size: Validation split proportion
        random_state: Random seed
        use_duration_list: List of boolean values for with/without duration. 
                          If None, defaults to [True, False]
        moves_n_list: List of move counts to test. If None, defaults to [None, 3, 5, 10]
        moves_only_n: Whether to keep exactly n moves (True) or create cumulative moves
        moves_new_column: Optional column name to store truncated moves
        moves_add_all_prefix: Prefix for cumulative move columns
    """
    # Default configurations
    if use_duration_list is None:
        use_duration_list = [True, False]
    
    if moves_n_list is None:
        moves_n_list = [None, 1, 2, 3, 4, 5]
    # Default: keep turns and drop turns
    if use_turns_list is None:
        use_turns_list = [True, False]
    # Default: keep victory_status and drop victory_status
    if use_victory_status_list is None:
        use_victory_status_list = [True, False]
    
    # Generate all combinations
    configurations = []
    for use_duration in use_duration_list:
        for use_turns in use_turns_list:
            for use_victory_status in use_victory_status_list:
                for moves_n in moves_n_list:
                    configurations.append({
                        "use_duration": use_duration,
                        "use_turns": use_turns,
                        "use_victory_status": use_victory_status,
                        "moves_n": moves_n,
                        "moves_only_n": moves_only_n,
                        "moves_new_column": moves_new_column,
                        "moves_add_all_prefix": moves_add_all_prefix,
                    })
    
    print(f"Running Optuna optimization for {len(configurations)} configurations:")
    for i, config in enumerate(configurations, 1):
        duration_str = "with duration" if config["use_duration"] else "without duration"
        turns_str = "with turns" if config["use_turns"] else "no turns"
        vstatus_str = "with vstatus" if config["use_victory_status"] else "no vstatus"
        moves_str = f"{config['moves_n']} moves" if config["moves_n"] is not None else "all moves"
        print(f"  {i}. {duration_str}, {turns_str}, {vstatus_str}, {moves_str}")
    print()
    
    results = []
    
    for i, config in enumerate(configurations, 1):
        duration_str = "duration" if config["use_duration"] else "noduration"
        turns_str = "withturn" if config["use_turns"] else "noturn"
        vstatus_str = "withvstatus" if config["use_victory_status"] else "novstatus"
        moves_str = "all_moves" if config["moves_n"] is None else f"{config['moves_n']}_moves"
        config_name = f"{duration_str}_{turns_str}_{vstatus_str}_{moves_str}"
        
        print(f"\n{'='*80}")
        print(f"Configuration {i}/{len(configurations)}: {config_name}")
        print(f"{'='*80}")
        print(f"  - Duration: {config['use_duration']}")
        print(f"  - Turns included: {config['use_turns']}")
        print(f"  - Victory status included: {config['use_victory_status']}")
        print(f"  - Moves: {config['moves_n']}")
        print(f"  - Only N moves: {config['moves_only_n']}")
        print()
        
        try:
            # Create a unique experiment name for each configuration
            config_experiment_name = f"{experiment_name}_{config_name}"
            
            study = run_study(
                csv_path=csv_path,
                experiment_name=config_experiment_name,
                n_trials=n_trials,
                test_size=test_size,
                random_state=random_state,
                use_duration=config["use_duration"],
                use_turns=config["use_turns"],
                use_victory_status=config["use_victory_status"],
                moves_n=config["moves_n"],
                moves_only_n=config["moves_only_n"],
                moves_new_column=config["moves_new_column"],
                moves_add_all_prefix=config["moves_add_all_prefix"],
            )
            
            best_value = study.best_value
            best_params = study.best_trial.params

            # Pull accuracy from study.best_metrics if present, else try metrics.json
            best_accuracy = None
            try:
                if hasattr(study, "best_metrics") and isinstance(study.best_metrics, dict):
                    best_accuracy = study.best_metrics.get("best_accuracy")
            except Exception:
                pass

            if best_accuracy is None:
                subdir = f"{duration_str}-{turns_str}-{('None' if config['moves_n'] is None else config['moves_n'])}-{bool(config['moves_only_n'])}"
                project_root = Path(__file__).resolve().parents[1]
                metrics_path = project_root / "models" / subdir / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            best_accuracy = m.get("best_accuracy")
                    except Exception:
                        pass

            results.append({
                "config_name": config_name,
                "use_duration": config["use_duration"],
                "moves_n": config["moves_n"],
                "use_turns": config["use_turns"],
                "use_victory_status": config["use_victory_status"],
                "moves_only_n": config["moves_only_n"],
                "best_f1_macro": best_value,
                "best_accuracy": best_accuracy,
                "best_params": best_params,
                "experiment_name": config_experiment_name,
            })
            
            print(f"\n✓ Completed: {config_name}")
            print(f"  Best F1 Macro: {best_value:.4f}")
            if best_accuracy is not None:
                try:
                    print(f"  Best Accuracy: {float(best_accuracy):.4f}")
                except Exception:
                    print(f"  Best Accuracy: {best_accuracy}")
            print(f"  Best Params: {best_params}")
            
        except Exception as e:
            print(f"\n✗ Error in configuration {config_name}: {str(e)}")
            results.append({
                "config_name": config_name,
                "use_duration": config["use_duration"],
                "moves_n": config["moves_n"],
                "use_turns": config["use_turns"],
                "moves_only_n": config["moves_only_n"],
                "error": str(e),
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        # Sort by best F1 score
        successful_results.sort(key=lambda x: x["best_f1_macro"], reverse=True)
        
        print("\nBest performing configurations (sorted by F1 Macro):")
        for i, result in enumerate(successful_results, 1):
            print(f"\n{i}. {result['config_name']}")
            print(f"   F1 Macro: {result['best_f1_macro']:.4f}")
            print(f"   Duration: {result['use_duration']}")
            print(f"   Turns included: {result['use_turns']}")
            print(f"   Victory status included: {result['use_victory_status']}")
            print(f"   Moves: {result['moves_n']}")
            print(f"   Best Params: {result['best_params']}")
    
    # Save results to JSON
    project_root = Path(__file__).resolve().parents[1]
    results_file = project_root / "models" / "multi_study_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save a concise accuracy log for easy reading
    log_file = project_root / "models" / "multi_study_accuracy.log"
    lines = [
        "config_name\tuse_duration\tuse_turns\tuse_victory_status\tmoves_n\tbest_accuracy\tbest_f1_macro"
    ]
    for r in results:
        lines.append(
            f"{r['config_name']}\t{r['use_duration']}\t{r.get('use_turns')}\t{r.get('use_victory_status')}\t{r['moves_n']}\t{r.get('best_accuracy')}\t{r.get('best_f1_macro')}"
        )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Accuracy log saved to: {log_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for multiple dataset configurations."
    )
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="./res/games.csv", 
        help="Path to games CSV"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="XGB_Optuna_Multi_Config", 
        help="Base MLflow experiment name"
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=20, 
        help="Number of Optuna trials per configuration"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.3, 
        help="Validation split proportion"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--use_duration", 
        type=str, 
        nargs="+", 
        default=None,
        help="List of duration options: 'true' for with duration, 'false' for without. "
             "Default: ['true', 'false']"
    )
    parser.add_argument(
        "--use_victory_status", 
        type=str, 
        nargs="+", 
        default=None,
        help="List of victory_status options: 'true' to keep 'victory_status' column, 'false' to drop it. "
             "Default: ['true', 'false']"
    )
    parser.add_argument(
        "--use_turns", 
        type=str, 
        nargs="+", 
        default=None,
        help="List of turns options: 'true' to keep 'turns' column, 'false' to drop it. "
             "Default: ['true', 'false']"
    )
    parser.add_argument(
        "--moves_n", 
        type=str, 
        nargs="+", 
        default=None,
        help="List of move counts to test. Use 'None' for all moves. Default: [None, 3, 5, 10]"
    )
    parser.add_argument(
        "--moves_only_n", 
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y"], 
        default=True, 
        help="If True, keep exactly n moves; else create cumulative moves_1..n"
    )
    parser.add_argument(
        "--moves_new_column", 
        type=str, 
        default=None, 
        help="Optional column name to store the truncated moves"
    )
    parser.add_argument(
        "--moves_add_all_prefix", 
        type=str, 
        default="moves_", 
        help="Prefix for cumulative move columns when moves_only_n is False"
    )
    
    args = parser.parse_args()
    
    # Parse use_duration_list
    use_duration_list = None
    if args.use_duration:
        use_duration_list = [
            str(v).lower() in ["1", "true", "yes", "y"] 
            for v in args.use_duration
        ]
    
    # Parse moves_n_list
    moves_n_list = None
    if args.moves_n:
        moves_n_list = []
        for v in args.moves_n:
            v_str = str(v).lower()
            if v_str == "none":
                moves_n_list.append(None)
            else:
                try:
                    moves_n_list.append(int(v))
                except ValueError:
                    raise ValueError(f"Invalid value for --moves_n: {v}. Must be an integer or 'None'")
    
    # Parse use_turns_list
    use_turns_list = None
    if args.use_turns:
        use_turns_list = [
            str(v).lower() in ["1", "true", "yes", "y"]
            for v in args.use_turns
        ]
    
    # Parse use_victory_status_list
    use_victory_status_list = None
    if args.use_victory_status:
        use_victory_status_list = [
            str(v).lower() in ["1", "true", "yes", "y"]
            for v in args.use_victory_status
        ]
    
    run_multi_study(
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        test_size=args.test_size,
        random_state=args.random_state,
        use_duration_list=use_duration_list,
        use_turns_list=use_turns_list,
        use_victory_status_list=use_victory_status_list,
        moves_n_list=moves_n_list,
        moves_only_n=args.moves_only_n,
        moves_new_column=args.moves_new_column,
        moves_add_all_prefix=args.moves_add_all_prefix,
    )

