import numpy as np
import pandas as pd

import analysis


def paths(mk, cd, start_hours, step_hours, liquid_only):
    start = analysis.snapshot(start_hours, mk, cd)
    start = start[start["live"] & (start["liquid"] if liquid_only else True)]
    after = cd[cd["hours_to_close"] < start_hours].copy()
    if liquid_only:
        after.loc[~(after["spread"] <= analysis.LIQUID), "mid"] = np.nan
    grid = np.round(after["hours_to_close"]) % step_hours == 0
    after = after[grid | (after["hours_to_close"] == 0)]
    after["mid"] = after.groupby("ticker")["mid"].ffill()

    by_ticker = {t: g["mid"].dropna().to_numpy() for t, g in after.groupby("ticker")}
    rows = []
    for r in start.itertuples():
        path = np.concatenate([[r.p], by_ticker.get(r.ticker, [])])
        rows.append({
            "ticker": r.ticker, "event_ticker": r.event_ticker, "category": r.category,
            "p0": r.p, "steps": len(path) - 1,
            "path": np.sum(np.diff(path) ** 2),
            "jump": (r.outcome - path[-1]) ** 2,
            "u0": r.p * (1 - r.p),
        })
    return pd.DataFrame(rows)


def movement_ratio(df, n_boot=analysis.N_BOOT):
    m, j, u = df["path"].to_numpy(), df["jump"].to_numpy(), df["u0"].to_numpy()

    def stat(_, w):
        total = (w * u).sum()
        return np.array([(w * m).sum() / total, (w * j).sum() / total,
                         (w * (m + j)).sum() / total])

    point, lo, hi = analysis.cluster_boot(df, stat, n_boot)
    return {"n": len(df), "events": df["event_ticker"].nunique(),
            "path": point[0], "jump": point[1],
            "total": point[2], "total_lo": lo[2], "total_hi": hi[2]}


def movement_table(mk, cd, starts=(144, 72), steps=(1, 6, 24)):
    rows = []
    for start in starts:
        for liquid_only in (False, True):
            for step in steps:
                df = paths(mk, cd, start, step, liquid_only)
                rows.append({"start_h": start, "quotes": "liquid" if liquid_only else "all",
                             "step_h": step, **movement_ratio(df)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    mk, cd = analysis.load()
    print("=== Realized movement / claimed uncertainty ===")
    print(movement_table(mk, cd).round(3).to_string(index=False))
