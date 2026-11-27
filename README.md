# Retail Cancellation Review

Retail Cancellation Review helps retail operations analysts decide which products to investigate for unusually high cancellation workload. It uses historical transactions to build a weekly review queue of up to 20 SKUs and shows what happened after each ranking.

- **Review queue** shows each flagged SKU’s rank, recent activity, and observed cancellation count for completed weeks.
- **Workload patterns** lets analysts explore weekly trends by market and identify the SKUs with the most cancellation documents.
- **Monitoring** shows data quality, queue coverage, and how the ranking performed against a simple historical baseline.

## Run locally

Use Python 3.12 from the repository root.

```sh
python3.12 -m venv .venv
PIP_CACHE_DIR="$PWD/.work/cache/pip" .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m retail_review
.venv/bin/python -m streamlit run app.py
```

Open <http://127.0.0.1:8501>. The analysis downloads public sources on the first run and reuses the snapshots on subsequent runs. It records their hashes and sizes in `.work/results.json`. The dashboard reads that file, and its server binds to loopback.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m retail_review --output .work/reproduced.json
cmp .work/results.json .work/reproduced.json
```

## Data and measures

- [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail), Daqing Chen, CC BY 4.0. The source covers December 2010 through December 2011 for one UK online retailer. Standard product codes, valid dates, and priced invoice lines determine scope. Cancellation documents have a cancellation prefix and negative quantity. Other adjustments are counted separately. Document-level aggregation counts a repeated SKU on one document once.
- [ECB daily GBP per EUR series](https://data.ecb.europa.eu/help/api/data-examples), series `EXR/D.GBP.EUR.SP00.A`. Each observed retail day can use the most recent strictly prior published rate within seven days. ECB holiday placeholders are counted separately. Exchange rates provide descriptive market context.
- The customer-level market comparison uses a two-sided Welch test on one cancellation-document share per identified customer. It compares UK customers with customers in countries that used the euro throughout the data window. IDs that appear in both groups are excluded. This comparison assumes independent customer observations and is sensitive to missing IDs and market composition.

## Review queue and evaluation

The task is to flag up to 20 SKUs on Monday where three or more distinct cancellation documents will be observed that week. The threshold and capacity are workload assumptions. A SKU must have at least one sale in the prior eight complete weeks. The first and last source weeks may be incomplete. The first week is omitted, and the last gets scores with outcomes pending.

Features count sales and cancellation documents over the preceding one, four, and eight weeks. Three expanding-window folds compare a four-week cancellation baseline, logistic regression, and shallow boosted trees. Mean weekly precision at 20 selects the method, with simpler methods preferred on ties. The final eight complete weeks are held out for evaluation. Monitoring reports weekly hits, eligible workload, precision, recall, and average precision. Weeks without positives contribute no recall or average precision value.

## Findings from the source snapshots

- The pipeline retained 536,429 of 541,909 retail lines. All 305 observed retail dates matched a prior ECB rate. Complete weeks contained 3,302 cancellation-marked documents. Workload peaked at 116 during the week starting December 13, 2010.
- Logistic regression led validation at 8.7% mean precision at 20, versus 8.3% for the baseline. On the untouched holdout, it flagged 17 high-workload SKU-weeks from 160 alerts, for 10.6% precision. The baseline flagged 19, for 11.9% precision. The baseline is the stronger operational starting point on this holdout. Both methods need more recent data before use for current decisions.
- The mean cancellation-document share was 1.55 percentage points higher among identified euro-area customers than UK customers. The 95% confidence interval ranged from −0.54 to +3.65 points and the two-sided p-value was 0.145. This comparison leaves substantial uncertainty about a market-specific difference.

Cancellation documents measure recorded workload. The source provides no confirmed reason codes, inventory availability, analyst review outcomes, or current data. Queue scores are for ranking. Reviewers can inspect historical evidence before deciding whether to investigate an SKU.
