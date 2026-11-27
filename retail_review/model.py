"""Build weekly, point-in-time features and evaluate review queues."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

QUEUE_SIZE = 20
HIGH_LOAD = 3
HOLDOUT_WEEKS = 8
FOLD_WEEKS = 5
FEATURES = ["sales_1w", "sales_4w", "sales_8w", "cancels_1w", "cancels_4w", "cancels_8w"]
MODELS = ("Recent cancellations", "Logistic regression", "Boosted trees")


def weekly_panel(lines: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range(lines["week"].min(), lines["week"].max(), freq="W-MON")
    if len(weeks) < 3:
        raise ValueError("Need at least one complete calendar week between the source boundaries")
    # The first and last calendar weeks can both be partial. Last week's features
    # are useful for scoring, but its target is intentionally withheld.
    observed = lines[lines["week"].isin(weeks[1:-1])]
    events = observed[["sku", "week", "invoice", "cancelled"]].drop_duplicates()
    weekly = events.groupby(["sku", "week"])["cancelled"].agg(["size", "sum"])
    weekly["sales"] = weekly["size"] - weekly["sum"]
    weekly["cancels"] = weekly["sum"]
    index = pd.MultiIndex.from_product([observed["sku"].unique(), weeks[1:]], names=["sku", "week"])
    panel = weekly[["sales", "cancels"]].reindex(index, fill_value=0).sort_index()
    for kind in ("sales", "cancels"):
        lag = panel.groupby(level="sku")[kind].shift(1)
        for window in (1, 4, 8):
            total = lag.groupby(level="sku").rolling(window, min_periods=window).sum()
            panel[f"{kind}_{window}w"] = total.droplevel(0)
    panel = panel.loc[panel["sales_8w"].gt(0)].reset_index()
    panel["cancel_docs"] = panel.pop("cancels").astype("Int64")
    panel.loc[panel["week"].eq(weeks[-1]), "cancel_docs"] = pd.NA
    panel["high_load"] = panel["cancel_docs"].ge(HIGH_LOAD)
    return panel.drop(columns="sales")


def score(model_name: str, train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    if model_name == "Recent cancellations":
        return rows["cancels_4w"].to_numpy() / 4
    if train["high_load"].nunique() != 2:
        raise ValueError("Training data must contain both target classes")
    if model_name == "Logistic regression":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=15, early_stopping=False, random_state=42,
        )
    with threadpool_limits(limits=2):
        estimator.fit(train[FEATURES], train["high_load"].astype(int))
        return estimator.predict_proba(rows[FEATURES])[:, 1]


def rank(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    ranked = rows[["week", "sku", "cancel_docs", "high_load", "cancels_4w", "sales_4w"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["week", "score", "sku"], ascending=[True, False, True])
    ranked["position"] = ranked.groupby("week").cumcount() + 1
    return ranked


def metrics(ranked: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    results = []
    for week, rows in ranked.groupby("week"):
        flagged = rows.head(QUEUE_SIZE)
        positives = int(rows["high_load"].sum())
        hits = int(flagged["high_load"].sum())
        results.append({
            "week": week, "eligible": len(rows), "positives": positives, "flagged": len(flagged),
            "hits": hits, "precision_at_20": hits / len(flagged),
            "recall_at_20": hits / positives if positives else None,
            "average_precision": float(average_precision_score(rows["high_load"].astype(int), rows["score"]))
            if positives else None,
        })
    weekly = pd.DataFrame(results)
    measures = ["precision_at_20", "recall_at_20", "average_precision"]
    mean = {key: float(weekly[key].mean()) if weekly[key].notna().any() else None for key in measures}
    return mean, weekly


def backtest(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    observed = panel.loc[panel["high_load"].notna()]
    weeks = sorted(observed["week"].unique())
    split = len(weeks) - HOLDOUT_WEEKS
    if split < 3 * FOLD_WEEKS + 8:
        raise ValueError("Need 31 eligible weeks for training, rolling validation, and holdout")
    history = observed[observed["week"].lt(weeks[split])]
    holdout = observed[observed["week"].ge(weeks[split])]
    validation = {}
    fold_dates = []
    for name in MODELS:
        predictions = []
        for start in range(split - 3 * FOLD_WEEKS, split, FOLD_WEEKS):
            start_week, end_week = weeks[start], weeks[start + FOLD_WEEKS]
            train = history[history["week"].lt(start_week)]
            review = history[history["week"].ge(start_week) & history["week"].lt(end_week)]
            if name == MODELS[0]:
                fold_dates.append({
                    "train_through": train["week"].max().date().isoformat(),
                    "review_from": start_week.date().isoformat(),
                    "review_through": review["week"].max().date().isoformat(),
                })
            predictions.append(rank(review, score(name, train, review)))
        validation[name], _ = metrics(pd.concat(predictions, ignore_index=True))
    selected = max(MODELS, key=lambda name: validation[name]["precision_at_20"])

    winners = rank(holdout, score(selected, history, holdout))
    selected_metrics, weekly = metrics(winners)
    baseline = rank(holdout, score(MODELS[0], history, holdout))
    baseline_metrics, base_weeks = metrics(baseline)
    weekly = weekly.merge(base_weeks[["week", "precision_at_20"]], on="week", suffixes=("", "_baseline"))
    upcoming = panel.loc[panel["high_load"].isna()]
    pending = rank(upcoming, score(selected, observed, upcoming)) if not upcoming.empty else upcoming
    queue = pd.concat([winners, pending], ignore_index=True) if not upcoming.empty else winners
    summary = {
        "model": selected, "queue_capacity": QUEUE_SIZE, "high_load_threshold": HIGH_LOAD,
        "validation": validation, "folds": fold_dates, "holdout": selected_metrics,
        "holdout_baseline": baseline_metrics, "holdout_from": weeks[split].date().isoformat(),
        "holdout_through": weeks[-1].date().isoformat(),
    }
    return summary, weekly, queue.loc[queue["position"].le(QUEUE_SIZE)]
