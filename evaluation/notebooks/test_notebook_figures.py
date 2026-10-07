"""Chart/data checks on synthetic fixtures; no project performance measured."""

import tempfile
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from notebook_figures import (finrl_feature_names, forecast_diagnostics,
                              plot_classification, plot_forecast_examples,
                              plot_shap_action, plot_trading, trading_comparison,
                              trading_metrics)


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dates = pd.bdate_range("2024-01-02", periods=3)
        self.accounts = pd.DataFrame({"date": self.dates, "account_value": [100, 105, 90]})
        self.prices = pd.DataFrame({"A": [10, 11, 9], "B": [20, 22, 18]}, index=self.dates)

    def tearDown(self):
        plt.close("all")






    def test_forecast_baseline_and_representative_cases(self):
        predictions = {}
        for ticker, error in [("A", 0), ("B", 1), ("C", 2)]:
            predictions[ticker] = pd.DataFrame({"actual": [10, 11, 12],
                                                "predicted": np.array([10, 11, 12]) + error,
                                                "previous_close": [9, 10, 11]}, index=self.dates)
        metrics = forecast_diagnostics(predictions)
        self.assertEqual(metrics.loc["A", "mae_improvement_pct"], 100)
        self.assertEqual(metrics.loc["B", "mae_improvement_pct"], 0)
        self.assertEqual(metrics.loc["C", "mae_improvement_pct"], -100)
        fig = plot_forecast_examples(predictions, metrics, self.root / "forecasts.png")
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue(fig.axes[0].get_title().startswith("C:"))



    def test_lstm_split_excludes_cross_boundary_targets_and_scaler_leakage(self):
        notebook = json.loads((Path(__file__).parent / "LSTM_ETH_Forecasting.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "LOOKBACK, HORIZON =" in "".join(cell["source"]))
        dates = pd.date_range("2022-10-01", "2025-02-01")
        prices = np.arange(len(dates), dtype=float) + 10
        # A large post-training level must never affect fitted scaling.
        prices[dates >= "2023-01-01"] += 10000
        frame = pd.DataFrame({name: prices for name in ["Open", "High", "Low", "Close", "Volume"]},
                             index=dates)
        scope = {"frame": frame, "FEATURES": list(frame.columns), "np": np, "pd": pd,
                 "MinMaxScaler": MinMaxScaler, "RUN_DIR": self.root, "display": lambda value: None}
        exec(source, scope)
        np.testing.assert_allclose(scope["scaler"].data_max_,
                                   frame.loc[frame.index < "2023-01-01"].max().to_numpy())
        targets = scope["dates"]
        self.assertTrue((targets[scope["train_mask"]] < np.datetime64("2023-01-01")).all())
        validation_dates = targets[scope["validation_mask"]]
        self.assertTrue((validation_dates >= np.datetime64("2023-01-01")).all())
        self.assertTrue((validation_dates < np.datetime64("2025-01-01")).all())
        self.assertTrue((targets[scope["test_mask"]] >= np.datetime64("2025-01-01")).all())
        # Six windows straddle each boundary and belong to no split.
        assigned = scope["train_mask"] | scope["validation_mask"] | scope["test_mask"]
        self.assertEqual((~assigned).sum(), 12)


if __name__ == "__main__":
    unittest.main()
