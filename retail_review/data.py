"""Load, validate, and summarize source records."""

import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import stats

RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/data.csv"
FX_URL = (
    "https://data-api.ecb.europa.eu/service/data/EXR/D.GBP.EUR.SP00.A"
    "?startPeriod=2010-11-26&endPeriod=2011-12-09&format=csvdata"
)
RETAIL_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]
EURO_AREA = {
    "Austria", "Belgium", "Cyprus", "EIRE", "Finland", "France", "Germany",
    "Greece", "Italy", "Malta", "Netherlands", "Portugal", "Spain", "Luxembourg",
    "Slovakia", "Slovenia",
}  # Euro-area membership throughout the observation window.


def fetch(url: str, path: Path, limit: int) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        with urlopen(url, timeout=60) as response, partial.open("wb") as output:
            remaining = limit
            while chunk := response.read(1024 * 1024):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError(f"Download exceeded {limit:,} bytes for {path.name}")
                output.write(chunk)
        shutil.move(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(source, "sha256").hexdigest()}


def load_raw(path: Path) -> pd.DataFrame:
    options = {"usecols": RETAIL_COLUMNS, "dtype": "string", "keep_default_na": False}
    try:
        return pd.read_csv(path, **options)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **options)


def prepare_retail(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lines = raw[RETAIL_COLUMNS].rename(columns={
        "InvoiceNo": "invoice", "StockCode": "sku", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "timestamp", "UnitPrice": "price",
        "CustomerID": "customer", "Country": "country",
    }).copy()
    for col in lines.columns:
        lines[col] = lines[col].str.strip()
    lines["invoice"] = lines["invoice"].str.upper()
    lines["sku"] = lines["sku"].str.upper()
    lines["customer"] = lines["customer"].str.replace(r"\.0$", "", regex=True)
    lines["timestamp"] = pd.to_datetime(lines["timestamp"], format="%m/%d/%Y %H:%M", errors="coerce")
    lines["quantity"] = pd.to_numeric(lines["quantity"], errors="coerce").astype(float)
    lines["price"] = pd.to_numeric(lines["price"], errors="coerce").astype(float)

    bad = (
        lines["timestamp"].isna() | ~np.isfinite(lines["quantity"]) | ~np.isfinite(lines["price"])
        | lines["invoice"].eq("") | lines["sku"].eq("")
    )
    product = lines["sku"].str.fullmatch(r"[0-9]{5}[A-Z]{0,2}")
    sale = lines["invoice"].str.fullmatch(r"[0-9]{6}") & lines["quantity"].gt(0)
    canc = lines["invoice"].str.fullmatch(r"C[0-9]{6}") & lines["quantity"].lt(0)
    regular = (sale | canc) & lines["price"].gt(0)
    reason = np.select(
        [bad.to_numpy(dtype=bool), (~product).to_numpy(dtype=bool), (~regular).to_numpy(dtype=bool)],
        ["invalid", "nonstandard_sku", "other_adjustment"], default="retained",
    )
    clean = lines.loc[reason == "retained"].copy()
    clean["cancelled"] = canc.loc[clean.index].astype(bool)
    clean["day"] = clean["timestamp"].dt.normalize()
    clean["week"] = clean["day"] - pd.to_timedelta(clean["day"].dt.dayofweek, unit="D")
    clean["market"] = np.select(
        [clean["country"].eq("United Kingdom").to_numpy(dtype=bool), clean["country"].isin(EURO_AREA)],
        ["UK", "Euro area"], default="Other",
    )
    quality = {key: int((reason == key).sum()) for key in ("retained", "invalid", "nonstandard_sku", "other_adjustment")}
    quality["input"] = len(raw)
    quality["retained_without_customer"] = int(clean["customer"].eq("").sum())
    quality["repeated_retained_rows"] = int(clean.duplicated().sum())
    # Repeated lines may be legitimate. Document-level aggregation handles them without guessing.
    return clean, quality


def add_fx(lines: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict]:
    fx = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]).rename(columns={
        "TIME_PERIOD": "fx_date", "OBS_VALUE": "gbp_per_eur",
    })
    fx["fx_date"] = pd.to_datetime(fx["fx_date"], format="%Y-%m-%d", errors="raise")
    fx["gbp_per_eur"] = pd.to_numeric(fx["gbp_per_eur"], errors="raise")
    input_count = len(fx)
    holidays = fx["OBS_STATUS"].eq("H") & fx["gbp_per_eur"].isna()
    fx = fx.loc[~holidays, ["fx_date", "gbp_per_eur"]]
    if (fx.empty or fx["fx_date"].duplicated().any()
            or not (np.isfinite(fx["gbp_per_eur"]) & fx["gbp_per_eur"].gt(0)).all()):
        raise ValueError("Expected one positive GBP/EUR observation per source date")
    daily = pd.DataFrame({"day": lines["day"].drop_duplicates().sort_values()})
    # Prior published business day avoids assuming a same-day rate was already available.
    daily = pd.merge_asof(
        daily, fx.sort_values("fx_date"), left_on="day", right_on="fx_date",
        direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=7),
    )
    joined = lines.merge(daily, on="day", how="left", validate="many_to_one")
    return joined, {
        "input": input_count, "published_rates": len(fx), "holiday_rows": int(holidays.sum()),
        "matched_retail_dates": int(daily["fx_date"].notna().sum()),
        "unmatched_retail_dates": int(daily["fx_date"].isna().sum()),
        "unmatched_retail_lines": int(joined["fx_date"].isna().sum()),
    }


def study_markets(lines: pd.DataFrame) -> dict:
    study = lines[lines["market"].isin(["UK", "Euro area"])].copy()
    has_customer = study["customer"].ne("")
    docs = study.loc[has_customer, ["customer", "market", "invoice", "cancelled"]].drop_duplicates()
    # Exclude customer IDs with inconsistent market labels in the source data.
    multiple = docs.groupby("customer")["market"].nunique().loc[lambda count: count.gt(1)].index
    docs = docs.loc[~docs["customer"].isin(multiple)]
    shares = docs.groupby(["market", "customer"])["cancelled"].mean()
    uk = shares.get("UK", pd.Series(dtype=float))
    eu = shares.get("Euro area", pd.Series(dtype=float))
    counts = {
        "uk_customers": len(uk), "euro_area_customers": len(eu),
        "uk_documents": int(docs["market"].eq("UK").sum()),
        "euro_area_documents": int(docs["market"].eq("Euro area").sum()),
        "unidentified_lines": int((~has_customer).sum()), "multi_market_customers": len(multiple),
    }
    if min(len(uk), len(eu)) < 2 or eu.var() + uk.var() == 0:
        return {**counts, "difference_pp": None, "interval_pp": None, "p_value": None}
    test = stats.ttest_ind(eu, uk, equal_var=False)
    ci = test.confidence_interval()
    return {
        **counts, "uk_mean_pct": float(uk.mean() * 100), "euro_area_mean_pct": float(eu.mean() * 100),
        "difference_pp": float((eu.mean() - uk.mean()) * 100),
        "interval_pp": [float(ci.low * 100), float(ci.high * 100)], "p_value": float(test.pvalue),
    }
