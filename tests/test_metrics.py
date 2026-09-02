import os

import numpy as np
import pytest
import scipy.stats as st

from src.experiment.metrics import calculate_aggregated_metrics, generate_summary_table, save_summary_to_csv


def _make_run_result(accuracy=90.0, f1=85.0, time_metric=10.0):
    return {
        "metrics": {"accuracy": accuracy, "f1_score": f1},
        "time_metric": time_metric,
        "best_val_metrics": {"accuracy": accuracy - 2, "f1_score": f1 - 2},
        "val_metrics_history": [{"accuracy": accuracy - 5}, {"accuracy": accuracy}],
    }


class TestCalculateAggregatedMetrics:
    def test_empty_input(self):
        """Empty run list should return empty dict."""
        assert calculate_aggregated_metrics([]) == {}

    def test_single_run(self):
        """Single run should have std=0 and mean equal to the value."""
        result = calculate_aggregated_metrics([_make_run_result()])
        assert result["runs_count"] == 1
        assert result["test_metrics"]["accuracy"]["mean"] == pytest.approx(90.0)
        assert result["test_metrics"]["accuracy"]["std"] == 0.0
        assert result["mean_time_s"] == pytest.approx(10.0)

    def test_multiple_runs(self):
        """Multiple runs should produce non-zero std and valid CI bounds."""
        runs = [_make_run_result(90, 85, 10), _make_run_result(92, 87, 12), _make_run_result(88, 83, 8)]
        result = calculate_aggregated_metrics(runs)

        assert result["runs_count"] == 3
        assert result["test_metrics"]["accuracy"]["mean"] == pytest.approx(90.0)
        assert result["test_metrics"]["accuracy"]["std"] > 0
        assert result["test_metrics"]["accuracy"]["ci_95_lower"] > 0
        assert result["test_metrics"]["accuracy"]["ci_95_upper"] > 0
        assert result["converged_runs"] == 3

    def test_no_convergence(self):
        """Run with time_metric=None should count as non-converged."""
        run = _make_run_result()
        run["time_metric"] = None
        result = calculate_aggregated_metrics([run])
        assert result["converged_runs"] == 0
        assert result["mean_time_s"] is None

    def test_partial_convergence(self):
        """Only runs with non-None time_metric should count as converged."""
        runs = [_make_run_result(time_metric=5.0), _make_run_result(time_metric=None)]
        result = calculate_aggregated_metrics(runs)
        assert result["converged_runs"] == 1
        assert result["mean_time_s"] == pytest.approx(5.0)

    def test_validation_metrics(self):
        """Validation metrics should be aggregated from best_val_metrics."""
        result = calculate_aggregated_metrics([_make_run_result()])
        assert "accuracy" in result["validation_metrics"]
        assert result["validation_metrics"]["accuracy"]["mean"] == pytest.approx(88.0)

    def test_formatted_string(self):
        """Formatted string should contain the mean value."""
        result = calculate_aggregated_metrics([_make_run_result()])
        fmt = result["test_metrics"]["accuracy"]["formatted"]
        assert "90.00" in fmt

    def test_ci_bounds(self):
        """CI lower bound should be <= mean and upper bound >= mean."""
        runs = [_make_run_result(90, 85, 10), _make_run_result(92, 87, 12)]
        result = calculate_aggregated_metrics(runs)
        acc = result["test_metrics"]["accuracy"]
        assert acc["ci_95_lower"] <= acc["mean"]
        assert acc["ci_95_upper"] >= acc["mean"]

    def test_single_run_ci_equals_mean(self):
        """With one run, CI bounds should collapse to the mean."""
        result = calculate_aggregated_metrics([_make_run_result(85.0)])
        acc = result["test_metrics"]["accuracy"]
        assert acc["ci_95_lower"] == pytest.approx(acc["mean"])
        assert acc["ci_95_upper"] == pytest.approx(acc["mean"])

    def test_identical_values_zero_std(self):
        """Identical values across runs should produce std=0 and CI=mean."""
        runs = [_make_run_result(90, 85, 10) for _ in range(3)]
        result = calculate_aggregated_metrics(runs)
        acc = result["test_metrics"]["accuracy"]
        assert acc["std"] == pytest.approx(0.0)
        assert acc["ci_95_lower"] == pytest.approx(acc["mean"])
        assert acc["ci_95_upper"] == pytest.approx(acc["mean"])

    def test_ci_mathematical_correctness(self):
        """CI computation should match manual scipy calculation."""
        values = [88.0, 90.0, 92.0]
        runs = [_make_run_result(v, 85, 10) for v in values]
        result = calculate_aggregated_metrics(runs)
        acc = result["test_metrics"]["accuracy"]

        expected_mean = np.mean(values)
        expected_sem = st.sem(values)
        expected_ci = st.t.interval(confidence=0.95, df=2, loc=expected_mean, scale=expected_sem)

        assert acc["mean"] == pytest.approx(expected_mean, rel=1e-6)
        assert acc["ci_95_lower"] == pytest.approx(max(0, expected_ci[0]), rel=1e-6)
        assert acc["ci_95_upper"] == pytest.approx(expected_ci[1], rel=1e-6)

    def test_ci_lower_clamped_to_zero(self):
        """CI lower bound should never go below 0."""
        runs = [_make_run_result(0.5, 0.3, 10), _make_run_result(0.1, 0.1, 10), _make_run_result(0.2, 0.2, 10)]
        result = calculate_aggregated_metrics(runs)
        acc = result["test_metrics"]["accuracy"]
        assert acc["ci_95_lower"] >= 0.0


class TestSaveSummaryToCsv:
    def test_saves_file(self, tmp_path):
        """Non-empty data should be written to a CSV file."""
        data = [
            {
                "experiment": "SGD_no_noise",
                "hyperparams": "lr=0.01",
                "epochs_num": 10,
                "mean_time_s": 5.0,
                "time_std_s": 1.0,
                "full_metrics": {
                    "accuracy": {"mean": 90.0, "std": 1.0, "ci_95_lower": 88.0, "ci_95_upper": 92.0},
                },
            }
        ]
        filepath = str(tmp_path / "summary.csv")
        save_summary_to_csv(data, filepath)
        assert os.path.exists(filepath)

        with open(filepath) as f:
            content = f.read()
        assert "SGD_no_noise" in content
        assert "accuracy_mean" in content

    def test_empty_data(self, tmp_path):
        """Empty data should not create a file."""
        filepath = str(tmp_path / "empty.csv")
        save_summary_to_csv([], filepath)
        assert not os.path.exists(filepath)


class TestGenerateSummaryTable:
    def test_prints_table(self, capsys):
        """Should print a formatted table with experiment data."""
        data = [
            {"Experiment": "SGD_no_noise", "Accuracy": "90.00 +/- 1.00"},
            {"Experiment": "Adam_no_noise", "Accuracy": "92.00 +/- 0.50"},
        ]
        generate_summary_table(data)
        captured = capsys.readouterr()
        assert "FINAL SUMMARY TABLE" in captured.out
        assert "SGD_no_noise" in captured.out

    def test_empty_data(self, capsys):
        """Empty data should still print the header."""
        generate_summary_table([])
        captured = capsys.readouterr()
        assert "FINAL SUMMARY TABLE" in captured.out
