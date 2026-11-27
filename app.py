"""Local review dashboard for the retail cancellation analysis."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).with_name(".work") / "results.json"


def percent(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.1%}"


def show_queue(result: dict) -> None:
    model = result["model"]
    queue = pd.DataFrame(result["queue"])
    queue["week"] = pd.to_datetime(queue["week"]).dt.date
    week = st.selectbox("Week starting Monday", sorted(queue["week"].unique(), reverse=True))
    rows = queue[queue["week"].eq(week)].copy()
    observed = rows["cancel_docs"].notna().all()
    matched = int(rows["high_load"].eq(True).sum())
    capacity, hits, threshold = st.columns(3)
    capacity.metric("SKUs flagged", f"{len(rows)}/{model['queue_capacity']}")
    hits.metric("Flagged SKUs with high workload", str(matched) if observed else "Pending")
    threshold.metric("High workload threshold", f"{model['high_load_threshold']} documents")
    if observed:
        st.caption("Historical replay. Each score uses activity before the selected Monday.")
        only_false_alerts = st.checkbox("Show only alerts below the observed threshold")
        if only_false_alerts:
            rows = rows.loc[rows["high_load"].eq(False)]
    else:
        st.caption("The source ends during this week. The queue uses prior activity. Outcomes need a complete week.")
    if rows.empty:
        st.info("No alerts match this filter.")
        return
    columns = ["position", "sku", "cancels_4w", "sales_4w", "score"]
    if observed:
        columns.append("cancel_docs")
    st.dataframe(
        rows[columns], hide_index=True,
        column_config={
            "position": "Priority", "sku": "SKU", "cancels_4w": "Prior 4 weeks · cancellations",
            "sales_4w": "Prior 4 weeks · sales documents",
            "score": st.column_config.NumberColumn("Ranking score", format="%.3f"),
            "cancel_docs": st.column_config.NumberColumn("Observed cancellation documents", format="%d"),
        },
    )
    st.caption(f"Ranked by {model['model'].lower()}. Scores order the queue and can share the same value.")


def show_patterns(result: dict) -> None:
    trends = pd.DataFrame(result["market_weeks"])
    trends["week"] = pd.to_datetime(trends["week"]).dt.date
    sku_counts = pd.DataFrame(result["sku_counts"])
    sku_counts["week"] = pd.to_datetime(sku_counts["week"]).dt.date
    market = st.selectbox("Market", ["All markets", *sorted(trends["market"].unique())])
    first, last = trends["week"].min(), trends["week"].max()
    start, end = st.slider("Week range", min_value=first, max_value=last, value=(first, last))
    trends = trends[trends["week"].between(start, end)]
    sku_counts = sku_counts[sku_counts["week"].between(start, end)]
    if market != "All markets":
        trends = trends[trends["market"].eq(market)]
        sku_counts = sku_counts[sku_counts["market"].eq(market)]
    if trends.empty:
        st.info("No transaction documents in this market and period.")
    else:
        weekly = trends.groupby("week", as_index=False)[["documents", "cancellations"]].sum()
        total = int(weekly["documents"].sum())
        cancellations = int(weekly["cancellations"].sum())
        docs, marked, share = st.columns(3)
        docs.metric("Transaction documents", f"{total:,}")
        marked.metric("Cancellation-marked documents", f"{cancellations:,}")
        share.metric("Cancellation-document share", f"{cancellations / total:.1%}")
        st.subheader("Weekly cancellation workload")
        st.line_chart(weekly, x="week", y="cancellations", x_label="Week", y_label="Documents")
        st.subheader("SKUs with the most cancellation documents")
        if sku_counts.empty:
            st.info("No SKU cancellations in this market and period.")
        else:
            leaders = sku_counts.groupby("sku")["cancellations"].sum().nlargest(10)
            chart = alt.Chart(leaders.reset_index()).mark_bar().encode(
                x=alt.X("cancellations:Q", title="Documents"),
                y=alt.Y("sku:N", sort="-x", title="SKU"),
                tooltip=["sku", "cancellations"],
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("A document can include several SKUs and contribute to each of their counts.")
        if market == "Euro area":
            st.subheader("Exchange-rate context")
            st.line_chart(trends, x="week", y="gbp_per_eur", x_label="Week", y_label="GBP per EUR")
            st.caption("Weekly mean of prior published ECB day rates across observed retail days.")
    st.subheader("Across eligible SKU-weeks")
    context = pd.DataFrame(result["historical_context"])
    context["Recent cancellation documents"] = pd.cut(
        context["cancels_4w"], bins=[-1, 0, 2, 5, float("inf")], labels=["0", "1–2", "3–5", "6+"],
    )
    context = context.groupby("Recent cancellation documents", as_index=False, observed=True)[
        ["sku_weeks", "high_load_weeks"]
    ].sum()
    context["Share with high workload"] = context["high_load_weeks"] / context["sku_weeks"]
    st.dataframe(context, hide_index=True, column_config={
        "sku_weeks": "Eligible SKU-weeks", "high_load_weeks": "High-workload SKU-weeks",
        "Share with high workload": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Prior four-week activity versus subsequent workload, across all markets and dates in the model dataset.")

    st.subheader("Customer-level market comparison")
    study = result["market_study"]
    if study["difference_pp"] is None:
        st.info("There were too few independent observations or no variation to run the test.")
        return
    low, high = study["interval_pp"]
    n_uk, n_eu = study["uk_customers"], study["euro_area_customers"]
    st.write(
        f"Euro-area customers averaged **{study['euro_area_mean_pct']:.2f}%** cancellation-marked documents "
        f"({n_eu:,} customers), and UK customers averaged **{study['uk_mean_pct']:.2f}%** ({n_uk:,} customers). "
        f"The difference was **{study['difference_pp']:+.2f} percentage points** "
        f"(95% CI {low:+.2f} to {high:+.2f}, Welch p = {study['p_value']:.3g})."
    )
    st.caption("Full dataset, identified customers only. Market comparisons are observational and can reflect customer mix.")


def show_monitoring(result: dict) -> None:
    model = result["model"]
    measured = model["holdout"]
    precision, recall, ap = st.columns(3)
    precision.metric("Mean precision at 20", percent(measured["precision_at_20"]))
    recall.metric("Mean recall at 20", percent(measured["recall_at_20"]))
    ap.metric("Mean average precision", percent(measured["average_precision"]))
    baseline = percent(model["holdout_baseline"]["precision_at_20"])
    st.caption(
        f"Holdout weeks starting {model['holdout_from']} through {model['holdout_through']}. "
        f"The recent-cancellation baseline averaged {baseline} precision at 20."
    )
    weekly = pd.DataFrame(result["performance"]).rename(columns={
        "precision_at_20": "Selected method", "precision_at_20_baseline": "Recent-cancellation baseline",
    })
    weekly["week"] = pd.to_datetime(weekly["week"]).dt.date
    st.subheader("Weekly precision at 20")
    st.line_chart(weekly, x="week", y=["Selected method", "Recent-cancellation baseline"], y_label="Precision")
    st.dataframe(weekly[["week", "eligible", "positives", "flagged", "hits"]], hide_index=True)
    st.caption(
        f"The queue was eligible to flag {model['eligible_high_load_sku_weeks']:,} of "
        f"{model['high_load_sku_weeks']:,} observed high-workload SKU-weeks in the holdout. "
        "Eligibility requires a sale in the prior eight weeks."
    )
    st.subheader("Input quality")
    retail, fx = result["quality"]["retail"], result["quality"]["fx"]
    st.write(
        f"The pipeline retained **{retail['retained']:,} of {retail['input']:,}** retail lines. "
        f"It excluded {retail['nonstandard_sku']:,} nonstandard product codes, "
        f"{retail['other_adjustment']:,} other adjustments, and {retail['invalid']:,} invalid lines."
    )
    st.write(
        f"The ECB source contained {fx['published_rates']:,} published rates and {fx['holiday_rows']:,} holiday rows. "
        f"The join matched {fx['matched_retail_dates']:,} retail dates and left {fx['unmatched_retail_dates']:,} unmatched."
    )
    with st.expander("Validation comparison"):
        validation = pd.DataFrame.from_dict(model["validation"], orient="index").rename_axis("Method")
        st.dataframe(validation.style.format("{:.1%}"))


st.set_page_config(page_title="Retail Cancellation Review", layout="wide")
st.title("Retail Cancellation Review")
st.caption("Prioritize SKUs for investigation and check how the queue performs on historical records.")
if not RESULTS.is_file():
    st.info("Generate the analysis with `python -m retail_review`, then reload this page.")
    st.stop()

result = json.loads(RESULTS.read_text(encoding="utf-8"))
st.caption(f"UK retail source records from {result['first_record']} through {result['last_record']}.")
queue_tab, patterns_tab, monitoring_tab = st.tabs(["Review queue", "Workload patterns", "Monitoring"])
with queue_tab:
    show_queue(result)
with patterns_tab:
    show_patterns(result)
with monitoring_tab:
    show_monitoring(result)
