import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "calibration-research"})

SLEEP = 0.15


CLASSIC_CATEGORIES = [
    "Politics", "Elections", "Economics", "Financials", "Companies", "World",
]
MIN_DURATION_HOURS = 48

BATCH_CANDLE_BUDGET = 9000


def get(path, params=None, retries=6):
    for attempt in range(retries):
        try:
            r = SESSION.get(f"{BASE}{path}", params=params, timeout=30)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  {type(e).__name__}, sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = 2 ** attempt
            print(f"  HTTP {r.status_code}, sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    raise RuntimeError(f"gave up on {path} after {retries} retries")


def markets_in_window(path, series_ticker, start_ts):
    params = {"series_ticker": series_ticker, "limit": 1000}
    if path == "/markets":
        params["status"] = "settled"
    cursor = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        data = get(path, p)
        rows = data.get("markets", [])
        yield from rows
        cursor = data.get("cursor")
        if not cursor or not rows:
            break
        oldest = rows[-1].get("settlement_ts") or rows[-1].get("close_time")
        if oldest and pd.Timestamp(oldest).timestamp() < start_ts:
            break


def get_cutoff_unix():
    data = get("/historical/cutoff")
    ts = data.get("market_settled_ts")
    if ts is None:
        raise RuntimeError(f"unexpected /historical/cutoff shape: {data}")
    if isinstance(ts, str):
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    return int(ts)


def series_for_categories(categories):
    out = {}
    for cat in categories:
        data = get("/series", {"category": cat})
        for s in data.get("series", []):
            out[s["ticker"]] = s.get("category", cat)
    print(f"{len(out)} series across {len(categories)} categories: {categories}")
    return out


MARKET_COLS = [
    "ticker", "event_ticker", "market_type", "yes_sub_title",
    "open_time", "close_time", "settlement_ts", "result",
    "last_price_dollars", "volume_fp", "open_interest_fp",
    "notional_value_dollars", "strike_type", "can_close_early", "category",
]


def build_market_frame(rows, min_duration_hours, start, end):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keep = [c for c in MARKET_COLS if c in df.columns]
    df = df[keep].copy()
    if "market_type" in df:
        df = df[df["market_type"] == "binary"]
    df = df[df["result"].isin(["yes", "no"])]
    df["outcome"] = (df["result"] == "yes").astype(int)
    df["series_ticker"] = df["ticker"].str.split("-").str[0]
    for col in ("open_time", "close_time", "settlement_ts"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("last_price_dollars", "volume_fp", "open_interest_fp"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open_time", "close_time"])
    settled = df["settlement_ts"].fillna(df["close_time"])
    df = df[(settled >= start) & (settled <= end)]
    dur_h = (df["close_time"] - df["open_time"]).dt.total_seconds() / 3600
    df = df[dur_h >= min_duration_hours]
    return df.drop_duplicates("ticker")


def collect_markets(start, end, categories, min_duration_hours):
    start_ts = int(start.timestamp())
    series_cat = series_for_categories(categories)

    path = OUT / "settled_markets.parquet"
    progress = OUT / "_markets_done.json"

    done = set(json.loads(progress.read_text())) if progress.exists() else set()
    survivors = [pd.read_parquet(path)] if done and path.exists() else []
    if done:
        print(f"resuming: {len(done)} series done, {len(survivors[0])} survivors on disk")

    def checkpoint():
        df = pd.concat(survivors, ignore_index=True) if survivors else pd.DataFrame()
        if not df.empty:
            df = df.drop_duplicates("ticker")
        df.to_parquet(path, index=False)
        progress.write_text(json.dumps(sorted(done)))
        return df

    for st, cat in series_cat.items():
        if st in done:
            continue
        raw = []
        for endpoint in ("/markets", "/historical/markets"):
            raw += markets_in_window(endpoint, st, start_ts)
        for m in raw:
            m["category"] = cat
        df_s = build_market_frame(raw, min_duration_hours, start, end)
        if not df_s.empty:
            survivors.append(df_s)
        done.add(st)
        if len(done) % 200 == 0:
            df = checkpoint()
            print(f"  {len(done)}/{len(series_cat)} series done, {len(df)} survivors")

    df = checkpoint()
    print(f"wrote {len(df)} settled binary markets (>= {min_duration_hours}h) -> {path}")
    return df


def candle_row(ticker, c):
    return {
        "ticker": ticker,
        "end_period_ts": c["end_period_ts"],
        "bid_close": (c.get("yes_bid") or {}).get("close_dollars"),
        "ask_close": (c.get("yes_ask") or {}).get("close_dollars"),
        "trade_close": (c.get("price") or {}).get("close_dollars"),
        "volume": c.get("volume_fp"),
        "open_interest": c.get("open_interest_fp"),
    }


def collect_candles(interval=60, horizon_days=7, sample=None):
    markets = pd.read_parquet(OUT / "settled_markets.parquet")
    if sample:
        markets = markets.sample(sample, random_state=0)

    cutoff = get_cutoff_unix()
    interval_sec = interval * 60

    cpath = OUT / "candles.parquet"
    dpath = OUT / "_candles_done.json"

    done_tk = set(json.loads(dpath.read_text())) if dpath.exists() else set()
    prior = pd.read_parquet(cpath) if done_tk and cpath.exists() else None
    if done_tk:
        print(f"resuming candles: {len(done_tk)} markets done, "
              f"{0 if prior is None else len(prior)} candles on disk")

    rows = []

    def flush():
        num_cols = ("bid_close", "ask_close", "trade_close", "volume", "open_interest")
        new = pd.DataFrame(rows, columns=["ticker", "end_period_ts", *num_cols])
        for col in num_cols:
            new[col] = pd.to_numeric(new[col], errors="coerce")
        new["mid"] = (new["bid_close"] + new["ask_close"]) / 2
        df = pd.concat([prior, new], ignore_index=True) if prior is not None else new
        df.to_parquet(cpath, index=False)
        dpath.write_text(json.dumps(sorted(done_tk)))
        return df

    m = markets[~markets["ticker"].isin(done_tk)].copy()
    if m.empty:
        df = flush()
        print(f"wrote {len(df)} candles -> {cpath}")
        return
    m["close_unix"] = m["close_time"].map(lambda x: int(x.timestamp()))
    m["settled_unix"] = (
        m["settlement_ts"].fillna(m["close_time"]).map(lambda x: int(x.timestamp()))
    )
    m["start_unix"] = m["close_unix"] - horizon_days * 86400
    hist = m[m["settled_unix"] < cutoff]
    live = m[m["settled_unix"] >= cutoff].sort_values("close_unix")

    for i, r in enumerate(hist.itertuples(), 1):
        try:
            data = get(
                f"/historical/markets/{r.ticker}/candlesticks",
                {"start_ts": r.start_unix, "end_ts": r.close_unix, "period_interval": interval},
            )
        except requests.HTTPError as e:
            print(f"  {r.ticker}: {e}")
            continue
        rows.extend(candle_row(r.ticker, c) for c in data.get("candlesticks", []))
        done_tk.add(r.ticker)
        if i % 100 == 0:
            print(f"historical {i}/{len(hist)} markets, {len(rows)} new candles")
            flush()

    lv = list(live.itertuples())
    i = processed = 0
    while i < len(lv):
        w_lo, w_hi, j = lv[i].start_unix, lv[i].close_unix, i + 1
        while j < len(lv):
            nlo, nhi = min(w_lo, lv[j].start_unix), max(w_hi, lv[j].close_unix)
            if (nhi - nlo) / interval_sec * (j - i + 1) > BATCH_CANDLE_BUDGET:
                break
            w_lo, w_hi, j = nlo, nhi, j + 1
        batch = lv[i:j]
        span = {r.ticker: (r.start_unix, r.close_unix) for r in batch}
        try:
            data = get(
                "/markets/candlesticks",
                {"market_tickers": ",".join(span),
                 "start_ts": w_lo, "end_ts": w_hi, "period_interval": interval},
            )
        except requests.HTTPError as e:
            print(f"  live batch at {i}: {e}")
            i = j
            continue
        for entry in data.get("markets", []):
            tk = entry.get("market_ticker")
            lo_hi = span.get(tk)
            if not lo_hi:
                continue
            for c in entry.get("candlesticks", []):
                if lo_hi[0] <= c["end_period_ts"] <= lo_hi[1]:
                    rows.append(candle_row(tk, c))
        done_tk.update(span)
        processed += len(batch)
        i = j
        if processed // 500 != (processed - len(batch)) // 500:
            print(f"live {processed}/{len(live)} markets, {len(rows)} new candles")
            flush()

    df = flush()
    print(f"wrote {len(df)} candles -> {cpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("markets")
    mp.add_argument("--start", required=True)
    mp.add_argument("--end", required=True)
    mp.add_argument("--categories", nargs="+", default=CLASSIC_CATEGORIES)
    mp.add_argument("--min-duration-hours", type=int, default=MIN_DURATION_HOURS)

    cp = sub.add_parser("candles")
    cp.add_argument("--interval", type=int, default=60, choices=[1, 60, 1440])
    cp.add_argument("--horizon-days", type=int, default=7)
    cp.add_argument("--sample", type=int, default=None,
                    help="random subsample of markets for a quick pass")

    args = ap.parse_args()
    if args.cmd == "markets":
        s = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        collect_markets(s, e, args.categories, args.min_duration_hours)
    else:
        collect_candles(args.interval, args.horizon_days, args.sample)
