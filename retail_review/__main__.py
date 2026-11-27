"""Generate a reproducible set of dashboard inputs from the source snapshots."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from .data import FX_URL, RETAIL_URL, add_fx, fetch, fingerprint, load_raw, prepare_retail, study_markets
from .model import HIGH_LOAD, backtest, weekly_panel


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="s"))


def summarize(lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = lines.loc[lines["week"].gt(lines["week"].min()) & lines["week"].lt(lines["week"].max())]
    documents = complete[["week", "market", "invoice", "cancelled"]].drop_duplicates()
    trends = documents.groupby(["week", "market"], as_index=False).agg(
        documents=("invoice", "size"), cancellations=("cancelled", "sum"),
    )
    trends["cancellation_share"] = trends["cancellations"] / trends["documents"]
    daily_fx = complete[["day", "week", "gbp_per_eur"]].drop_duplicates()
    by_week = daily_fx.groupby("week", as_index=False)["gbp_per_eur"].mean()
    trends = trends.merge(by_week, on="week", how="left", validate="many_to_one")
    cancellations = complete.loc[complete["cancelled"], ["week", "market", "sku", "invoice"]].drop_duplicates()
    # Keep all SKU counts so the dashboard can rank correctly within any date window.
    sku_counts = cancellations.groupby(["week", "market", "sku"], as_index=False).agg(
        cancellations=("invoice", "size"),
    )
    return trends, sku_counts


def build(raw_dir: Path, output: Path) -> dict:
    retail = raw_dir / "retail.csv"
    fx = raw_dir / "fx.csv"
    fetch(RETAIL_URL, retail, limit=68_000_000)
    fetch(FX_URL, fx, limit=1_000_000)
    lines, retail_quality = prepare_retail(load_raw(retail))
    if lines.empty:
        raise ValueError("No valid product lines in the retail snapshot")
    lines, fx_quality = add_fx(lines, fx)
    trends, sku_counts = summarize(lines)
    panel = weekly_panel(lines)
    model, weekly, queue = backtest(panel)

    in_holdout = lines["week"].between(pd.Timestamp(model["holdout_from"]), pd.Timestamp(model["holdout_through"]))
    actual = lines.loc[in_holdout & lines["cancelled"], ["week", "sku", "invoice"]].drop_duplicates()
    high_load = actual.groupby(["week", "sku"]).size().ge(HIGH_LOAD)
    model["high_load_sku_weeks"] = int(high_load.sum())
    model["eligible_high_load_sku_weeks"] = int(weekly["positives"].sum())
    context = panel.loc[panel["high_load"].notna()].groupby("cancels_4w", as_index=False).agg(
        sku_weeks=("high_load", "size"), high_load_weeks=("high_load", "sum"),
    )
    bundle = {
        "quality": {"retail": retail_quality, "fx": fx_quality},
        "sources": {"retail": fingerprint(retail), "fx": fingerprint(fx)},
        "first_record": lines["day"].min().date().isoformat(),
        "last_record": lines["day"].max().date().isoformat(),
        "market_study": study_markets(lines),
        "model": model,
        "performance": records(weekly),
        "queue": records(queue),
        "market_weeks": records(trends),
        "sku_counts": records(sku_counts),
        "historical_context": records(context),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as temporary:
        draft = Path(temporary.name)
        try:
            json.dump(bundle, temporary, allow_nan=False, separators=(",", ":"))
            temporary.close()
            draft.replace(output)
        finally:
            draft.unlink(missing_ok=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(".work/raw"))
    parser.add_argument("--output", type=Path, default=Path(".work/results.json"))
    args = parser.parse_args()
    bundle = build(args.raw_dir, args.output)
    retail = bundle["quality"]["retail"]
    model = bundle["model"]
    print(f"Retained {retail['retained']:,} of {retail['input']:,} retail lines")
    print(f"Selected {model['model']} with {model['holdout']['precision_at_20']:.1%} mean holdout precision at 20")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
