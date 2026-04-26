from typing import Any, Optional

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from tqdm import tqdm

from src.config import load_config
from src.data.processing import get_dataloaders
from src.training.evaluate import evaluate_model
from src.utils import SPLIT_RANDOM_STATE


def find_best_model_uri(experiment_name: str, optimizer_name: str, scenario_name: str) -> Optional[str]:
    """
    Finds the URI of the best model by searching through the run history.

    Can work with:
    1. save_mode='best' (searches for the model in the Parent Run using the best_run_val_accuracy metric).
    2. save_mode='all' (searches for the best Child Run).
    3. Situations where the last run crashed or produced a poor result
       (the function will automatically search previous runs).
    """
    print(f"Searching for best model for '{optimizer_name}_{scenario_name}'...")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    parent_run_name = f"{optimizer_name}_{scenario_name}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        order_by=["attribute.start_time DESC"],
    )

    if not runs:
        print(f"No runs found with name '{parent_run_name}'.")
        return None

    print(f"  - Found {len(runs)} historical runs. Checking for valid artifacts...")

    for run in runs:
        run_id = run.info.run_id
        metrics = run.data.metrics

        if "best_run_val_accuracy" in metrics:
            acc = metrics["best_run_val_accuracy"]
            print(f"  - Found valid Parent Run (ID: {run_id}) with save_mode='best'.")
            print(f"    Validation Accuracy: {acc:.2f}%")
            return f"runs:/{run_id}/best_model_across_runs"

        child_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            order_by=["metrics.final_val_accuracy DESC"],
            max_results=1,
        )

        if child_runs:
            best_child = child_runs[0]
            child_acc = best_child.data.metrics.get("final_val_accuracy", 0)
            if child_acc > 0:
                print(f"  - Found valid Child Run (ID: {best_child.info.run_id}) in Parent ({run_id}).")
                print(f"    Validation Accuracy: {child_acc:.2f}%")
                return f"runs:/{best_child.info.run_id}/final_model"

    print(f"Could not find ANY valid model for '{parent_run_name}' in history.")
    return None


def run_comparative_robustness_evaluation(config_path: str, optimizer_registry: dict[str, Any]) -> None:
    """Evaluates and compares the robustness of the best models across all noise scenarios."""
    config = load_config(config_path)

    experiment_name = config.mlflow.experiment_name.format(
        model_name=config.model.name,
        dataset_name=config.data.dataset_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizers_to_compare = list(optimizer_registry.keys())
    scenario_trained_on = config.robustness.trained_on_scenario

    print(f"Starting comparative robustness evaluation for optimizers: {optimizers_to_compare}")

    all_results: list[dict[str, Any]] = []

    for optimizer_name in optimizers_to_compare:
        print("\n" + "=" * 80)
        print(f"PROCESSING OPTIMIZER: {optimizer_name}")
        print("=" * 80)

        model_uri = find_best_model_uri(
            experiment_name=experiment_name,
            optimizer_name=optimizer_name,
            scenario_name=scenario_trained_on,
        )

        if not model_uri:
            print(f"Skipping '{optimizer_name}' as no valid champion model was found.")
            continue

        print(f"Loading model from URI: {model_uri}...")
        try:
            model = mlflow.pytorch.load_model(model_uri, map_location=device)
            model.eval()
        except Exception as e:
            print(f"ERROR: Failed to load model for '{optimizer_name}'. Skipping. Error: {e}")
            continue

        for scenario_name in tqdm(config.grid_search.noise_scenarios, desc=f"Evaluating {optimizer_name}"):
            try:
                _, _, test_loader = get_dataloaders(
                    preprocessed_root_path=config.data.preprocessed_root_path,
                    scenario_folder_template=config.data.scenario_folder_template,
                    scenario_name=scenario_name,
                    random_state=SPLIT_RANDOM_STATE,
                    batch_size=config.training.batch_size,
                    num_workers=config.data.num_workers,
                    pin_memory=config.data.pin_memory,
                    subset_size=config.data.debug_subset_size,
                )
                with torch.no_grad():
                    metrics = evaluate_model(model, test_loader, device)

                row = {"optimizer": optimizer_name, "test_scenario": scenario_name}
                row.update(metrics)
                all_results.append(row)
            except Exception as e:
                print(f"ERROR: Failed scenario '{scenario_name}' for '{optimizer_name}': {e}")

    if not all_results:
        print("\nNo results were collected. Cannot generate a summary table.")
        return

    results_df = pd.DataFrame(all_results)

    try:
        pivot_df = results_df.pivot_table(index="test_scenario", columns="optimizer", values="accuracy")
        desired_order = list(config.grid_search.noise_scenarios.keys())
        pivot_df = pivot_df.reindex([s for s in desired_order if s in pivot_df.index])

        print("\n\n" + "=" * 80)
        print(" " * 15 + "COMPARATIVE ROBUSTNESS SUMMARY (Metric: accuracy)")
        print("=" * 80)

        pd.set_option("display.width", 1000)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.precision", 2)
        print(pivot_df)
        print("=" * 80)
    except Exception as e:
        print(f"Failed to create a pivot table: {e}. Displaying raw data:")
        print(results_df)

    results_df.to_csv("comparative_robustness_evaluation.csv", index=False, float_format="%.4f")
    print("\nFull results saved to 'comparative_robustness_evaluation.csv'")
