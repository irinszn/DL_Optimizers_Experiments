from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st
from tabulate import tabulate


def calculate_aggregated_metrics(run_results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculates aggregated metrics over a list of results.

    Returns:
        A dictionary with the full structure of calculated metrics. Example:
        {
            'runs_count': 3,
            'validation_metrics': { ... },
            'test_metrics': {
                'accuracy': {
                    'mean': 92.33,
                    'ci_95': (91.4, 93.2),
                    'formatted': "92.33 (91.40, 93.20)"
                },
                'f1_score': { ... }
            },
            'mean_time_s': 111.23
        }
    """
    if not run_results_list:
        return {}

    num_runs = len(run_results_list)
    aggregated_results: dict[str, Any] = {
        "runs_count": num_runs,
        "validation_metrics": {},
        "test_metrics": {},
        "mean_time_s": None,
        "val_metrics_history_per_run": [
            run["val_metrics_history"] for run in run_results_list if run.get("val_metrics_history")
        ],
    }

    best_val_metrics_list = [run["best_val_metrics"] for run in run_results_list if run.get("best_val_metrics")]
    if best_val_metrics_list:
        for key in best_val_metrics_list[0].keys():
            values = [d.get(key) for d in best_val_metrics_list if d.get(key) is not None]
            if values:
                aggregated_results["validation_metrics"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values) if num_runs > 1 else 0.0,
                }

    test_metrics: list[dict[str, float]] = [run["metrics"] for run in run_results_list if run.get("metrics")]
    if test_metrics:
        for key in test_metrics[0].keys():
            values = [d.get(key) for d in test_metrics if d.get(key) is not None]
            if not values:
                continue

            mean_val = np.mean(values)
            ci_95 = (mean_val, mean_val)
            if num_runs > 1:
                sem = st.sem(values)
                if sem > 0:
                    ci_95 = st.t.interval(confidence=0.95, df=num_runs - 1, loc=mean_val, scale=sem)

            aggregated_results["test_metrics"][key] = {
                "mean": mean_val,
                "ci_95_lower": max(0, ci_95[0]),
                "ci_95_upper": ci_95[1],
                "std": np.std(values) if num_runs > 1 else 0.0,
                "formatted": f"{mean_val:.2f} ({max(0, ci_95[0]):.2f}, {ci_95[1]:.2f})",
            }

    times: list[float] = [run["time_metric"] for run in run_results_list if run.get("time_metric") is not None]
    aggregated_results["converged_runs"] = len(times)
    if times:
        aggregated_results["mean_time_s"] = np.mean(times)
        aggregated_results["time_std_s"] = np.std(times) if len(times) > 1 else 0.0

    return aggregated_results


def generate_summary_table(data: list[dict]) -> None:
    """Prints the resulting pivot table to the console."""
    print("\n\n" + "=" * 100)
    print(" " * 40 + "FINAL SUMMARY TABLE")
    print("=" * 100)

    if not data:
        print("There is no data to display in the summary.")
        return

    summary_df = pd.DataFrame(data)
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False, numalign="center", stralign="center"))


def save_summary_to_csv(summary_data: list[dict], filename: str = "experiment_summary.csv") -> None:
    """Saves a complete summary of experiments to a CSV file."""
    if not summary_data:
        print("There is no data to save to CSV.")
        return

    records = []
    for row in summary_data:
        record = {
            "experiment": row["experiment"],
            "hyperparams": row["hyperparams"],
            "epochs_num": row["epochs_num"],
            "conv_time_mean_s": row["mean_time_s"],
            "conv_time_std_s": row["time_std_s"],
        }
        for metric_name, data in row["full_metrics"].items():
            record[f"{metric_name}_mean"] = data.get("mean")
            record[f"{metric_name}_std"] = data.get("std")
            record[f"{metric_name}_ci95_lower"] = data.get("ci_95_lower")
            record[f"{metric_name}_ci95_upper"] = data.get("ci_95_upper")
        records.append(record)

    try:
        pd.DataFrame(records).to_csv(filename, index=False, float_format="%.2f")
        print(f"\nThe full summary was successfully saved to file: {filename}")
    except Exception as e:
        print(f"\nError saving CSV file: {e}")
