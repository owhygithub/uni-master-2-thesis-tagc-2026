"""
fetch_fundamentals_sec.py

Fetches historical quarterly fundamentals directly from SEC EDGAR.
Free, no API key, no rate limits (just polite throttling).

Fixes in this version:
  - Expanded XBRL concept fallbacks (fixes no-usable-data for GS, MSFT, CAT etc.)
  - revenue/net_income NOT forward-filled (flow concepts — ffill would repeat
    annual values into quarterly rows, distorting profit_margin)
  - Only balance sheet items (total_debt, total_equity) are forward-filled
  - total_debt zero-filled where missing and equity exists (genuinely zero debt)
  - Removed LiabilitiesAndStockholdersEquity from equity fallbacks (it includes
    liabilities, giving wrong D/E ratio)

Output:
  data/fundamentals_quarterly.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data"
START_DATE = pd.Timestamp("2012-01-01")
USER_AGENT = "Oskar valens.valenswot@gmail.com"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load tickers ──────────────────────────────────────────────────────────────
TICKER_FILE = "stocks.txt"
with open(TICKER_FILE) as f:
    TICKERS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Loaded {len(TICKERS)} tickers from {TICKER_FILE}")

HEADERS = {"User-Agent": USER_AGENT}

# ── Step 1: CIK mapping ───────────────────────────────────────────────────────
print("Loading SEC ticker→CIK mapping...")
r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
r.raise_for_status()
ticker_to_cik = {
    v["ticker"].upper(): str(v["cik_str"]).zfill(10)
    for v in r.json().values()
}

# ── Step 2: Fetch company facts ──────────────────────────────────────────────
def fetch_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Step 3: Concept fallbacks ─────────────────────────────────────────────────
CONCEPT_FALLBACKS = {
    "revenue": [
        # Standard
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        # Financials / banks (GS, MS, AXP, V, MA)
        "RevenuesNetOfInterestExpense",
        "InterestAndDividendIncomeOperating",
        "NoninterestIncome",
        # Utilities (NEE, DUK, SO)
        "RegulatedAndUnregulatedOperatingRevenue",
        "ElectricUtilityRevenue",
        # Industrials / diversified (CAT, EMR, HON, HD, TGT)
        "SalesRevenueGoodsNet",
        "TotalRevenuesAndOtherIncome",
        "OtherIncome",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLossAttributableToParent",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_debt": [
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "NotesPayable",
        "SeniorNotes",
        "DebtCurrent",
        "ShortTermBorrowings",
    ],
    "total_equity": [
        # Only true equity concepts — NOT LiabilitiesAndStockholdersEquity
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
        "MembersEquity",
        "RetainedEarningsAccumulatedDeficit",   # last resort
    ]
}

# ── Step 4: Extract concept ───────────────────────────────────────────────────
def extract_concept(facts, concept_keys, is_balance_sheet=False):
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    master = {}   # merged results across all fallback concepts
    for key in concept_keys:
        if key not in us_gaap:
            continue
        units = us_gaap[key].get("units", {})
        unit_key = next(iter(units), None)
        if not unit_key:
            continue

        out = {}
        for e in units[unit_key]:
            form  = e.get("form", "")
            end   = e.get("end")
            val   = e.get("val")
            filed = e.get("filed")
            start = e.get("start")

            if not end or val is None or not filed:
                continue
            if form not in ("10-Q", "10-K"):
                continue

            # Period-length filter for flow concepts only.
            # 10-Q: no length filter — 180-day gap filter is sufficient.
            #        52-week fiscal companies (AAPL, MSFT, COST) have inconsistent
            #        XBRL start dates causing valid quarters to fail length checks.
            # 10-K: must be annual (330-400 days) to exclude stub periods.
            if not is_balance_sheet and start and form == "10-K":
                try:
                    period_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
                    if not (330 <= period_days <= 400):
                        continue
                except Exception:
                    continue

            existing = out.get(end)
            if existing is None:
                out[end] = {"val": val, "filed": filed, "form": form}
                continue
            if existing["form"] == "10-K" and form == "10-Q":
                out[end] = {"val": val, "filed": filed, "form": form}
                continue
            if existing["form"] == "10-Q" and form == "10-K":
                continue
            if filed < existing["filed"]:
                out[end] = {"val": val, "filed": filed, "form": form}

        # Merge into master dict — first concept to cover a date wins,
        # subsequent concepts fill in missing dates only
        for date, info in out.items():
            if date not in master:
                master[date] = info

    if master:
        return ({d: i["val"]   for d, i in master.items()},
                {d: i["filed"] for d, i in master.items()},
                {d: i["form"]  for d, i in master.items()})
    return {}, {}, {}

# ── Step 5: Build per-ticker frame ────────────────────────────────────────────
def build_ticker_frame(ticker, facts):
    rev_vals,    _,        _        = extract_concept(facts, CONCEPT_FALLBACKS["revenue"])
    ni_vals,     ni_filed, ni_form  = extract_concept(facts, CONCEPT_FALLBACKS["net_income"])
    debt_vals,   _,        _        = extract_concept(facts, CONCEPT_FALLBACKS["total_debt"],   is_balance_sheet=True)
    equity_vals, _,        _        = extract_concept(facts, CONCEPT_FALLBACKS["total_equity"], is_balance_sheet=True)

    # Anchor rows on net_income dates
    anchor_dates = sorted(ni_vals.keys())
    if not anchor_dates:
        return pd.DataFrame()

    rows = []
    for date in anchor_dates:
        net_income = ni_vals.get(date)

        # Revenue: exact match first, then nearest date within 10 days
        # Apple and other 52-week fiscal year companies sometimes have
        # revenue XBRL dates 1-2 days off from net_income dates
        revenue = rev_vals.get(date)
        if revenue is None and rev_vals:
            close = min(rev_vals, key=lambda d: abs((pd.to_datetime(d) - pd.to_datetime(date)).days))
            if abs((pd.to_datetime(close) - pd.to_datetime(date)).days) <= 10:
                revenue = rev_vals[close]

        # Balance sheet: match nearest date within 10 days
        debt = debt_vals.get(date)
        if debt is None and debt_vals:
            close = min(debt_vals, key=lambda d: abs((pd.to_datetime(d) - pd.to_datetime(date)).days))
            if abs((pd.to_datetime(close) - pd.to_datetime(date)).days) <= 10:
                debt = debt_vals[close]

        equity = equity_vals.get(date)
        if equity is None and equity_vals:
            close = min(equity_vals, key=lambda d: abs((pd.to_datetime(d) - pd.to_datetime(date)).days))
            if abs((pd.to_datetime(close) - pd.to_datetime(date)).days) <= 10:
                equity = equity_vals[close]

        rows.append({
            "ticker":            ticker,
            "fiscal_period_end": date,
            "reported_date":     ni_filed.get(date),
            "form":              ni_form.get(date),
            "revenue":           revenue,
            "net_income":        net_income,
            "total_debt":        debt,
            "total_equity":      equity,
        })

    df = pd.DataFrame(rows)
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df["reported_date"]     = pd.to_datetime(df["reported_date"])
    df = df[df["fiscal_period_end"] >= START_DATE].sort_values("fiscal_period_end").reset_index(drop=True)

    if df.empty:
        return df

    # ── Derived ratios (before any filling) ──────────────────────────────────
    df["profit_margin"]  = np.where(
        df["revenue"].notna() & (df["revenue"] != 0),
        df["net_income"] / df["revenue"], np.nan
    )
    df["debt_to_equity"] = np.where(
        df["total_equity"].notna() & (df["total_equity"] != 0),
        df["total_debt"] / df["total_equity"], np.nan
    )
    df["roe"] = np.where(
        df["total_equity"].notna() & (df["total_equity"] != 0),
        df["net_income"] / df["total_equity"], np.nan
    )

    # ── Cleaning ──────────────────────────────────────────────────────────────
    # 1. Dedup
    df = df.drop_duplicates(subset=["fiscal_period_end"], keep="first")

    # 2. Forward-fill ONLY balance sheet items (point-in-time, safe to carry forward)
    #    Do NOT ffill revenue/net_income — these are flow concepts; repeating an
    #    annual value into quarterly rows would corrupt profit_margin
    bs_ffill = ["total_debt", "total_equity", "debt_to_equity"]
    df[bs_ffill] = df[bs_ffill].ffill()

    # 3. Zero-fill total_debt where still missing but equity exists
    #    (company genuinely carries no debt — EDGAR simply has no filing for it)
    df["total_debt"] = df["total_debt"].fillna(0)
    df["debt_to_equity"] = np.where(
        df["total_equity"].notna() & (df["total_equity"] != 0),
        df["total_debt"] / df["total_equity"], df["debt_to_equity"]
    )


    # 5. Winsorize ratios at 1st/99th percentile
    for col in ["profit_margin", "debt_to_equity", "roe"]:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # 5b. Fill remaining NaN ratios with 0 (negative equity edge cases)
    #     Negative equity is economically meaningful but NaN breaks model training
    for col in ["debt_to_equity", "roe"]:
        df[col] = df[col].fillna(0)

    # 6. Drop rows missing core flow concepts (can't be imputed)
    df = df.dropna(subset=["net_income", "revenue", "total_equity"])

    # 7. Reorder
    cols = ["ticker", "fiscal_period_end", "reported_date", "form",
            "revenue", "net_income",
            "total_debt", "total_equity",
            "profit_margin", "debt_to_equity", "roe"]
    return df[[c for c in cols if c in df.columns]]

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nFetching SEC EDGAR fundamentals for {len(TICKERS)} tickers...\n")
frames = []

for i, ticker in enumerate(TICKERS):
    cik = ticker_to_cik.get(ticker.upper())
    if not cik:
        print(f"[{i+1}/{len(TICKERS)}] {ticker}  [!] CIK not found, skipping")
        continue

    print(f"[{i+1}/{len(TICKERS)}] {ticker} (CIK {cik})")
    try:
        facts = fetch_company_facts(cik)
        df = build_ticker_frame(ticker, facts)
        if not df.empty:
            frames.append(df)
            print(f"  -> {len(df)} quarters | nulls: {df.isnull().sum().to_dict()}")
        else:
            print(f"  -> no usable data")
    except requests.HTTPError as e:
        print(f"  [!] HTTP {e.response.status_code}")
    except Exception as e:
        print(f"  [!] {e}")

    time.sleep(0.15)

# ── Save ──────────────────────────────────────────────────────────────────────
if not frames:
    print("\n[!] No data collected.")
else:
    df_all = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "fundamentals_quarterly.csv")
    df_all.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
    print(f"  {len(df_all)} rows | {df_all['ticker'].nunique()} tickers | {len(df_all.columns)} columns")
    print(f"\nNull counts:\n{df_all.isnull().sum()}")
    print(f"\nSample:\n{df_all.head(8).to_string(index=False)}")

    print(df_all)