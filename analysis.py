import warnings

import numpy as np
import pandas as pd

DATA = "data"
BAND = (0.10, 0.90)
LIQUID = 0.10
BINS = np.arange(0, 1.05, 0.05)
HORIZONS = [(144, "6d"), (72, "3d"), (24, "1d"), (6, "6h"), (1, "1h"), (0, "close")]
N_BOOT = 1000

KALSHI_TABLE1 = {
    "Economics":              {"1wk": 0.089, "1d": 0.073, "1h": 0.066, "close": 0.066},
    "Elections":              {"1wk": 0.033, "1d": 0.022, "1h": 0.006, "close": 0.004},
    "Financials":             {"1wk": 0.069, "1d": 0.085, "1h": 0.032, "close": 0.032},
    "Politics":               {"1wk": 0.069, "1d": 0.053, "1h": 0.016, "close": 0.014},
    "Science and Technology": {"1wk": 0.056, "1d": 0.035, "1h": 0.016, "close": 0.015},
}


def load():
    mk = pd.read_parquet(f"{DATA}/settled_markets.parquet")
    cd = pd.read_parquet(f"{DATA}/candles.parquet")
    mk["close_unix"] = (mk["close_time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    mk["event_volume"] = mk.groupby("event_ticker")["volume_fp"].transform("sum")
    cd = cd.merge(mk[["ticker", "close_unix"]], on="ticker")
    cd["hours_to_close"] = (cd["close_unix"] - cd["end_period_ts"]) / 3600
    cd["spread"] = cd["ask_close"] - cd["bid_close"]
    return mk, cd.sort_values(["ticker", "end_period_ts"])


def snapshot(horizon_hours, mk=None, cd=None):
    if mk is None:
        mk, cd = load()
    eligible = cd[(cd["hours_to_close"] >= horizon_hours) & cd["mid"].notna()]
    last = eligible.groupby("ticker").tail(1)
    df = mk.merge(last[["ticker", "mid", "spread", "hours_to_close"]], on="ticker")
    df = df[(df["mid"] > 0) & (df["mid"] < 1)].copy()
    df["p"] = df["mid"]
    df["live"] = df["p"].between(*BAND)
    df["liquid"] = df["spread"] <= LIQUID
    df["horizon"] = horizon_hours
    return df


def cluster_boot(df, stat, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    codes, events = pd.factorize(df["event_ticker"])
    k = len(events)
    point = np.asarray(stat(df, np.ones(len(df))))
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[codes].astype(float)
        v = stat(df, w)
        if v is not None:
            draws.append(v)
    if not draws:
        return point, np.full_like(point, np.nan), np.full_like(point, np.nan)
    draws = np.asarray(draws)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return point, np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def _murphy(df, w):
    p, y = df["p"].to_numpy(), df["outcome"].to_numpy().astype(float)
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    n = np.bincount(idx, weights=w, minlength=len(BINS) - 1)
    ok = n > 0
    pk = np.bincount(idx, weights=w * p, minlength=len(BINS) - 1)[ok] / n[ok]
    yk = np.bincount(idx, weights=w * y, minlength=len(BINS) - 1)[ok] / n[ok]
    total, ybar = w.sum(), (w * y).sum() / w.sum()
    return np.array([
        (w * (p - y) ** 2).sum() / total,
        (n[ok] * (pk - yk) ** 2).sum() / total,
        (n[ok] * (yk - ybar) ** 2).sum() / total,
        ybar * (1 - ybar),
    ])


def score(df, n_boot=N_BOOT):
    point, lo, hi = cluster_boot(df, _murphy, n_boot)
    out = {"n": len(df), "events": df["event_ticker"].nunique()}
    for i, name in enumerate(["brier", "rel", "res"]):
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = point[i], lo[i], hi[i]
    out["unc"] = point[3]
    return out


def reliability_table(df, n_boot=N_BOOT):
    idx = pd.cut(df["p"], BINS, include_lowest=True)
    tab = df.groupby(idx, observed=True).agg(
        n=("outcome", "size"), mean_price=("p", "mean"), freq_yes=("outcome", "mean"),
        events=("event_ticker", "nunique"),
    ).reset_index(names="bin")
    codes = idx.cat.codes.to_numpy()
    y = df["outcome"].to_numpy().astype(float)

    def freq(_, w):
        with np.errstate(invalid="ignore"):
            return (np.bincount(codes, weights=w * y, minlength=len(BINS) - 1)
                    / np.bincount(codes, weights=w, minlength=len(BINS) - 1))

    _, lo, hi = cluster_boot(df, freq, n_boot)
    present = np.sort(np.unique(codes))
    tab["ci_lo"], tab["ci_hi"] = lo[present], hi[present]
    return tab


def _logit_fit(x, y, w):
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        grad = X.T @ (w * (y - mu))
        hess = X.T @ (X * (w * mu * (1 - mu))[:, None])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def logistic(df, n_boot=N_BOOT):
    x = np.log(df["p"] / (1 - df["p"])).to_numpy()
    y = df["outcome"].to_numpy().astype(float)

    def fit(_, w):
        m = w > 0
        try:
            return _logit_fit(x[m], y[m], w[m])
        except np.linalg.LinAlgError:
            return None

    point, lo, hi = cluster_boot(df, fit, n_boot)
    return {"n": len(df), "events": df["event_ticker"].nunique(),
            "a": point[0], "a_lo": lo[0], "a_hi": hi[0],
            "b": point[1], "b_lo": lo[1], "b_hi": hi[1]}


def horizon_table(mk, cd):
    snaps = {h: snapshot(h, mk, cd) for h, _ in HORIZONS}
    cohort = set(snaps[144]["ticker"])
    rows = []
    for h, label in HORIZONS:
        d = snaps[h]
        for sample, sub in [("per-horizon", d), ("cohort", d[d["ticker"].isin(cohort)]),
                            ("cohort, 2-98", d[d["ticker"].isin(cohort) & d["p"].between(0.02, 0.98)]),
                            ("cohort, live", d[d["ticker"].isin(cohort) & d["live"]])]:
            rows.append({"horizon": label, "sample": sample, **score(sub)})
    return pd.DataFrame(rows)


def category_table(mk, cd):
    ours = {label: snapshot(h, mk, cd) for h, label in HORIZONS}
    rows = []
    for cat, theirs in KALSHI_TABLE1.items():
        for label, key in [("6d", "1wk"), ("1d", "1d"), ("1h", "1h"), ("close", "close")]:
            d = ours[label][ours[label]["category"] == cat]
            s = score(d)
            rows.append({"category": cat, "horizon": label, "n": s["n"], "brier": s["brier"],
                         "lo": s["brier_lo"], "hi": s["brier_hi"], "kalshi": theirs[key]})
    return pd.DataFrame(rows)


def volume_table(df, edges=(0, 1e4, 5e4, 2e5, np.inf)):
    labels = ["<10K", "10-50K", "50-200K", ">=200K"]
    bucket = pd.cut(df["event_volume"], edges, labels=labels)
    rows = []
    for b, g in df.groupby(bucket, observed=True):
        rows.append({"bucket": b, "sample": "all", "live_share": g["live"].mean(), **score(g)})
        rows.append({"bucket": b, "sample": "live", "live_share": 1.0, **score(g[g["live"]])})
    return pd.DataFrame(rows)


def spread_table(live, cuts=(1.00, 0.40, 0.20, 0.10, 0.05)):
    rows = []
    for c in cuts:
        d = live[live["spread"] <= c]
        rows.append({"max_spread": c, **logistic(d), "rel": score(d)["rel"]})
    return pd.DataFrame(rows)


def lookahead_audit(mk, cd, horizon_hours=24):
    df = snapshot(horizon_hours, mk, cd)
    later = cd[(cd["hours_to_close"] < horizon_hours) & cd["mid"].between(*BAND)]
    leaky = df["live"] | df["ticker"].isin(set(later["ticker"]))
    assert df.loc[df["live"], "p"].between(*BAND).all()
    return {
        "n_live": int(df["live"].sum()),
        "n_leaky": int(leaky.sum()),
        "b_live": logistic(df[df["live"]], n_boot=0)["b"],
        "b_leaky": logistic(df[leaky], n_boot=0)["b"],
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    mk, cd = load()
    d24 = snapshot(24, mk, cd)
    live = d24[d24["live"]]
    liquid = live[live["liquid"]]

    print("=== Sample ===")
    print(f"markets {len(mk)}, candles {len(cd)}, with 24h quote {len(d24)}, "
          f"events {d24['event_ticker'].nunique()}, live {len(live)}, live+liquid {len(liquid)}")

    print("\n=== Brier by horizon (Kalshi Fig. 2/3 with a decomposition) ===")
    print(horizon_table(mk, cd).round(4).to_string(index=False))

    print("\n=== Brier by category vs Kalshi Table 1 ===")
    print(category_table(mk, cd).round(3).to_string(index=False))

    print("\n=== Brier by event volume at 24h (Kalshi Table 2) ===")
    print(volume_table(d24).round(4).to_string(index=False))

    print("\n=== Favorite-longshot fit vs spread cut, live at 24h ===")
    print(spread_table(live).round(3).to_string(index=False))

    print("\n=== Logistic fit by horizon, live and live+liquid ===")
    for h, label in HORIZONS[:4]:
        d = snapshot(h, mk, cd)
        lv = d[d["live"]]
        for name, sub in [("live", lv), ("live+liquid", lv[lv["liquid"]])]:
            f = logistic(sub)
            print(f"  {label:>3} {name:12s} n={f['n']:5d}  b={f['b']:.2f} ({f['b_lo']:.2f}-{f['b_hi']:.2f})"
                  f"  a={f['a']:.2f} ({f['a_lo']:.2f}-{f['a_hi']:.2f})")

    print("\n=== Reliability, live+liquid at 24h ===")
    print(reliability_table(liquid).round(3).to_string(index=False))

    print("\n=== Look-ahead audit ===")
    print(lookahead_audit(mk, cd))
