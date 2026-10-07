"""Focused figures for corrected notebooks. No training or fabricated results.

Dependencies: numpy, pandas, matplotlib, scikit-learn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)


def trading_comparison(account_frame, price_frame, initial_amount, buy_cost=0.0):
    """Align actual portfolio and equal-weight hold by date, without truncation.

    Benchmark buys fractional shares at first-date closes, paying buy_cost once,
    then marks to market without a final liquidation. The initial observation is
    pre-trade capital for both strategies; the first later observation includes
    the benchmark entry fee, matching FinRL's initial asset-memory convention.
    No ongoing rebalancing.
    """
    if initial_amount <= 0 or not 0 <= buy_cost < 1:
        raise ValueError("Positive initial capital and a buy cost in [0, 1) required")
    frame = account_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate account dates: use canonical environment asset memory")
    portfolio = frame.set_index("date")["account_value"].sort_index().astype(float)
    prices = price_frame.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    if prices.index.has_duplicates or prices.columns.has_duplicates:
        raise ValueError("Benchmark dates and ticker columns must be unique")
    if len(portfolio) < 2 or prices.shape[1] == 0:
        raise ValueError("At least two dates and one benchmark ticker required")
    if not portfolio.index.equals(prices.index):
        raise ValueError("Portfolio and benchmark date sets differ; fix data alignment, don't truncate")
    if not np.isfinite(portfolio).all() or (portfolio <= 0).any():
        raise ValueError("Portfolio values must be finite and positive")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Benchmark prices must be complete, finite, and positive")
    if not np.isclose(portfolio.iloc[0], initial_amount):
        raise ValueError("Portfolio must include the initial capital observation")
    shares = initial_amount / prices.shape[1] / (prices.iloc[0] * (1 + buy_cost))
    benchmark = prices.mul(shares, axis="columns").sum(axis=1)
    benchmark.iloc[0] = initial_amount
    return pd.DataFrame({"PPO": portfolio, "Equal-weight hold": benchmark})


def trading_metrics(comparison, initial_amount, periods_per_year=252):
    rows = []
    for name, values in comparison.items():
        # Both initial observations are pre-trade capital; the first subsequent
        # return includes any entry fees, matching the portfolio convention.
        returns = values.pct_change().dropna()
        deviation = returns.std(ddof=1)
        peaks = values.cummax().clip(lower=initial_amount)
        rows.append({"strategy": name, "return_pct": (values.iloc[-1] / initial_amount - 1) * 100,
                     "max_drawdown_pct": (values / peaks - 1).min() * 100,
                     "annualized_volatility_pct": deviation * np.sqrt(periods_per_year) * 100,
                     "sharpe_rf_zero": returns.mean() / deviation * np.sqrt(periods_per_year)
                     if len(returns) > 1 and deviation > 0 else np.nan})
    return pd.DataFrame(rows).set_index("strategy")


def plot_trading(comparison, initial_amount, output):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    colors = {"PPO": "#14326e", "Equal-weight hold": "#c36b25"}
    for name, values in comparison.items():
        axes[0].plot(values.index, (values / initial_amount - 1) * 100,
                     label=name, color=colors.get(name), linewidth=1.6)
        drawdown = (values / values.cummax().clip(lower=initial_amount) - 1) * 100
        axes[1].plot(values.index, drawdown, label=name, color=colors.get(name), linewidth=1.4)
    axes[0].set(title="Held-out return relative to initial capital", ylabel="Return (%)")
    axes[1].set(title="Drawdown from running peak", ylabel="Drawdown (%)", xlabel="Test date")
    for ax in axes:
        ax.axhline(0, color="#777777", linewidth=0.6)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def plot_classification(y_true, probabilities, output, positive_label, threshold=0.5):
    labels = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or len(labels) == 0:
        raise ValueError("One probability per held-out row is required")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation needs both original binary classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    report_dict = classification_report(labels, predictions, output_dict=True, zero_division=0)
    accuracy = report_dict.pop("accuracy")
    report = pd.DataFrame(report_dict).T
    summary = {"held_out_rows": len(labels), "positive_prevalence": float(labels.mean()),
               "average_precision": float(average_precision_score(labels, probabilities)),
               "roc_auc": float(roc_auc_score(labels, probabilities)), "threshold": threshold,
               "accuracy": accuracy}
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].step(recall, precision, where="post", color="#14326e",
                 label=f"Average precision = {summary['average_precision']:.3f}")
    axes[0].axhline(labels.mean(), linestyle="--", color="#888888", label="Positive prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
                title=f"Held-out {positive_label}: precision–recall")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center",
                         color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], xlabel="Predicted class", ylabel="Actual class",
                title=f"Original test rows; threshold {threshold:g}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return summary, report, fig


def finrl_feature_names(tickers, indicators):
    # FinRL's actual state is cash, prices, holdings, then indicator-major blocks.
    return (["cash"] + [f"{ticker}: close" for ticker in tickers]
            + [f"{ticker}: shares" for ticker in tickers]
            + [f"{ticker}: {indicator}" for indicator in indicators for ticker in tickers])


def plot_shap_action(values, feature_names, ticker, output):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(feature_names) or len(values) == 0:
        raise ValueError("Expected SHAP values shaped [samples, features] for ONE stock action")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite")
    means = np.abs(values).mean(axis=0)
    top = np.argsort(means)[-min(10, len(feature_names)):]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh([feature_names[i] for i in top], means[top], color="#14326e")
    ax.set(title=f"Inputs influencing the {ticker} policy action on sampled test states",
           xlabel="Mean absolute SHAP value (policy action units)")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def forecast_diagnostics(predictions):
    """One-step price errors against previous-close persistence, per ticker."""
    rows = []
    for ticker, frame in predictions.items():
        required = ["actual", "predicted", "previous_close"]
        values = frame[required].to_numpy(dtype=float)
        if len(frame) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Empty or nonfinite predictions for {ticker}")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
            raise ValueError(f"Unique target dates required for {ticker}")
        actual, predicted, baseline = values.T
        model_mae = np.abs(actual - predicted).mean()
        baseline_mae = np.abs(actual - baseline).mean()
        scale = np.abs(actual).mean()
        rows.append({"ticker": ticker, "test_rows": len(actual), "model_mae": model_mae,
                     "persistence_mae": baseline_mae,
                     "model_rmse": np.sqrt(np.mean((actual - predicted) ** 2)),
                     "persistence_rmse": np.sqrt(np.mean((actual - baseline) ** 2)),
                     "model_mae_pct_mean_price": 100 * model_mae / scale if scale else np.nan,
                     "persistence_mae_pct_mean_price": 100 * baseline_mae / scale if scale else np.nan,
                     "mae_improvement_pct": 100 * (1 - model_mae / baseline_mae)
                     if baseline_mae else np.nan})
    return pd.DataFrame(rows).set_index("ticker")


def plot_forecast_examples(predictions, metrics, output):
    ordered = metrics["mae_improvement_pct"].dropna().sort_values()
    if ordered.empty:
        raise ValueError("No nonzero persistence MAE available for example selection")
    indices = sorted(set([0, len(ordered) // 2, len(ordered) - 1]))
    selected = [ordered.index[i] for i in indices]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.5 * len(selected)),
                             squeeze=False, layout="constrained")
    for ticker, ax in zip(selected, axes[:, 0]):
        example = predictions[ticker].sort_index().tail(50)
        ax.plot(example.index, example["actual"], color="#222222", label="Actual")
        ax.plot(example.index, example["previous_close"], color="#aaaaaa", label="Previous-close baseline")
        ax.plot(example.index, example["predicted"], color="#14326e", label="Transformer")
        ax.set(title=f"{ticker}: test MAE improvement {ordered[ticker]:+.1f}% (positive is better)",
               ylabel="Close price (USD)", xlabel="Held-out target date")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Worst, median, and best baseline-relative cases; last 50 test observations", fontsize=10)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig
