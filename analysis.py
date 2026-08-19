import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")


def load(horizon_hours=24):
    mk = pd.read_parquet(DATA / "settled_markets.parquet")
    cd = pd.read_parquet(DATA / "candles.parquet")

    mk["close_unix"] = (
        mk["close_time"] - pd.Timestamp(0, tz="UTC")
    ) // pd.Timedelta("1s")
    cd = cd.merge(mk[["ticker", "close_unix"]], on="ticker", how="inner")
    cd["hours_to_close"] = (cd["close_unix"] - cd["end_period_ts"]) / 3600

    eligible = cd[(cd["hours_to_close"] >= horizon_hours) & cd["mid"].notna()]
    snap = (
        eligible.sort_values("end_period_ts")
        .groupby("ticker", as_index=False)
        .tail(1)
    )

    df = mk.merge(
        snap[["ticker", "mid", "bid_close", "ask_close", "volume", "hours_to_close"]],
        on="ticker",
        how="inner",
    )
    n0 = len(df)
    df = df.dropna(subset=["mid"])
    df = df[(df["mid"] > 0) & (df["mid"] < 1)]
    print(
        f"horizon={horizon_hours}h: {len(df)} markets with two-sided quotes "
        f"({n0 - len(df)} dropped for missing/degenerate quotes)"
    )
    df["p"] = df["mid"]
    df["spread"] = df["ask_close"] - df["bid_close"]
    return df


def reliability_table(df, bins=None):
    if bins is None:
        bins = np.arange(0, 1.05, 0.05)
    df = df.copy()
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)

    tab = df.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        mean_price=("p", "mean"),
        freq_yes=("outcome", "mean"),
        n_events=("event_ticker", "nunique"),
    )

    lo, hi = _cluster_bootstrap_bins(df, bins)
    tab["ci_lo"], tab["ci_hi"] = lo, hi
    return tab.reset_index()


def _cluster_bootstrap_bins(df, bins, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    events = df["event_ticker"].unique()
    bin_idx = pd.cut(df["p"], bins, include_lowest=True).cat.codes.to_numpy()
    outcome = df["outcome"].to_numpy()
    ev_codes = pd.Categorical(df["event_ticker"], categories=events).codes

    n_bins = len(bins) - 1
    boot = np.full((n_boot, n_bins), np.nan)
    for b in range(n_boot):
        sampled = rng.integers(0, len(events), len(events))
        counts = np.bincount(sampled, minlength=len(events))  # cluster weights
        w = counts[ev_codes].astype(float)
        num = np.bincount(bin_idx, weights=w * outcome, minlength=n_bins)
        den = np.bincount(bin_idx, weights=w, minlength=n_bins)
        with np.errstate(invalid="ignore"):
            boot[b] = num / den
    return (
        np.nanpercentile(boot, 2.5, axis=0),
        np.nanpercentile(boot, 97.5, axis=0),
    )


def brier(df, bins=None):
    p, y = df["p"].to_numpy(), df["outcome"].to_numpy()
    bs = np.mean((p - y) ** 2)
    if bins is None:
        bins = np.arange(0, 1.05, 0.05)
    idx = pd.cut(df["p"], bins, include_lowest=True)
    g = df.groupby(idx, observed=True)
    nk = g.size().to_numpy()
    pk = g["p"].mean().to_numpy()
    yk = g["outcome"].mean().to_numpy()
    ybar = y.mean()
    n = len(y)
    rel = np.sum(nk * (pk - yk) ** 2) / n
    res = np.sum(nk * (yk - ybar) ** 2) / n
    unc = ybar * (1 - ybar)
    return {"brier": bs, "reliability": rel, "resolution": res,
            "uncertainty": unc, "check(REL-RES+UNC)": rel - res + unc}


def flb_regression(df):
    x = np.log(df["p"] / (1 - df["p"])).to_numpy()
    y = df["outcome"].to_numpy()
    a, b = _logit_fit(x, y)

    rng = np.random.default_rng(0)
    events = df["event_ticker"].unique()
    ev_codes = pd.Categorical(df["event_ticker"], categories=events).codes
    boots = []
    for _ in range(1000):
        sampled = rng.integers(0, len(events), len(events))
        counts = np.bincount(sampled, minlength=len(events))
        w = counts[ev_codes].astype(float)
        mask = w > 0
        try:
            boots.append(_logit_fit(x[mask], y[mask], w[mask]))
        except Exception:
            continue
    boots = np.array(boots)
    return {
        "intercept": a,
        "slope": b,
        "intercept_ci": tuple(np.percentile(boots[:, 0], [2.5, 97.5])),
        "slope_ci": tuple(np.percentile(boots[:, 1], [2.5, 97.5])),
    }


def _logit_fit(x, y, w=None, iters=50):
    if w is None:
        w = np.ones_like(x, dtype=float)
    beta = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        grad = X.T @ (w * (y - mu))
        W = w * mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def sliced_brier(df, col, q=4):
    df = df.copy()
    df["slice"] = pd.qcut(df[col].rank(method="first"), q, labels=False)
    out = []
    for s, g in df.groupby("slice"):
        row = {"slice": s, "n": len(g), f"{col}_median": g[col].median()}
        row.update(brier(g))
        row["flb_slope"] = flb_regression(g)["slope"]
        out.append(row)
    return pd.DataFrame(out)


def by_category(df, min_n=200):
    out = []
    for cat, g in df.groupby("category"):
        if len(g) < min_n:
            continue
        row = {"category": cat, "n": len(g)}
        row.update(brier(g))
        out.append(row)
    return pd.DataFrame(out).sort_values("reliability", ascending=False)


def naive_edge_after_costs(df, fee_rate=0.07):
    tab = reliability_table(df)
    tab["gross_edge"] = tab["freq_yes"] - tab["mean_price"]
    tab["fee"] = fee_rate * tab["mean_price"] * (1 - tab["mean_price"])
    half_spread = df.groupby(
        pd.cut(df["p"], np.arange(0, 1.05, 0.05), include_lowest=True),
        observed=True,
    )["spread"].median().to_numpy() / 2
    tab["half_spread"] = half_spread
    tab["net_edge_buy_yes"] = tab["gross_edge"] - tab["fee"] - tab["half_spread"]
    tab["net_edge_buy_no"] = -tab["gross_edge"] - tab["fee"] - tab["half_spread"]
    return tab


if __name__ == "__main__":
    df = load(horizon_hours=24)
    print("\n=== Brier decomposition (24h horizon) ===")
    print(brier(df))
    print("\n=== Favorite-longshot regression ===")
    print(flb_regression(df))
    print("\n=== By volume quartile ===")
    print(sliced_brier(df, "volume_fp" if "volume_fp" in df else "volume"))
    print("\n=== Edge after costs ===")
    print(naive_edge_after_costs(df)[
        ["bin", "n", "mean_price", "freq_yes", "net_edge_buy_yes", "net_edge_buy_no"]
    ].to_string())
