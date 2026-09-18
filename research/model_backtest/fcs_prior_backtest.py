"""
Where should non-FBS teams start?

Production seeds every team missing from the preseason prior at 0.0 -- the same
number as an average FBS team -- and FCS opponents mostly are missing (SP+ only
covers FBS, and last season's Massey fit only sees their one or two games against
FBS teams). Nearly half of the games in the first few weeks are FBS-vs-FCS, so an
FCS team that starts out rated like a mid-pack FBS side makes a routine 35-point
win look like a strong result and inflates the FBS team that got it.

This replays 2022-2025 week by week and swaps only how non-FBS teams are seeded
(and, for some variants, whether FCS-vs-FCS games are in the fit). Everything is
graded on FBS-vs-FBS games, the only ones the site publishes.

Variants
  prod            Production as it is: train on games involving an FBS team, last
                  season's Massey fit on the same, FCS teams filled with 0.
  fixed_<c>       As prod, but every FCS team starts at c points relative to the
                  average FBS prior.
  unified         "Same methodology": last season's Massey fit on every FBS and
                  FCS game (FCS-vs-FCS included) on one scale, so an FCS team's
                  prior is its own rating from last season. Mirrors the FBS blend:
                  half own rating, half the FCS average (FBS teams use half SP+).
  unified_all     unified, plus FCS-vs-FCS games from this season in the fit.
  fixed_<c>_all   fixed_<c>, plus FCS-vs-FCS games in the fit.

Leakage control matches early_backtest.py: SP+ is the prior season's final SP+,
last-year ratings use only last season's games, week W is fit on weeks < W.
"""
import os

import numpy as np
import pandas as pd

import models as M
from backtest import grade_pick

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEASONS = [2022, 2023, 2024, 2025]
EARLY_WEEKS = {1, 2, 3, 4}
SP_WEIGHT = 0.5
HFA = 2.5
LAM = 3.0
FIXED_OFFSETS = [-10, -15, -20, -25, -30]


def load_games(year):
    df = pd.DataFrame(M.load_json(f"games_{year}.json"))
    df = df[df["seasonType"] == "regular"].dropna(subset=["homePoints", "awayPoints"]).copy()
    df["neutralSite"] = df["neutralSite"].fillna(False).astype(bool)
    df["hc"] = df["homeClassification"].fillna("")
    df["ac"] = df["awayClassification"].fillna("")
    return df.reset_index(drop=True)


def divisions(df):
    """Team -> classification that season, read off the games themselves."""
    out = {}
    for side in ("home", "away"):
        for t, c in zip(df[f"{side}Team"], df[f"{side[0]}c"]):
            if c:
                out[t] = c
    return out


def involves_fbs(df):
    return (df["hc"] == "fbs") | (df["ac"] == "fbs")


def fbs_or_fcs_only(df):
    return df["hc"].isin(["fbs", "fcs"]) & df["ac"].isin(["fbs", "fcs"])


def massey(games):
    """Plain least-squares margin ratings, home field fit, centred on all teams."""
    r, _ = M.fit_massey_ratings(games)
    return r


def ridge_to_prior(train, prior):
    """Same objective as production: ||y - Xb||^2 + lam*||b - prior||^2, HFA fixed.
    Teams missing from the prior start at 0, exactly as production does."""
    if len(train) == 0:
        return prior.copy()
    teams = sorted(set(train["homeTeam"]) | set(train["awayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(train), len(teams)))
    rows = np.arange(len(train))
    X[rows, train["homeTeam"].map(idx).values] = 1.0
    X[rows, train["awayTeam"].map(idx).values] = -1.0
    hfa = np.where(train["neutralSite"].values, 0.0, HFA)
    y = (train["homePoints"] - train["awayPoints"]).values - hfa
    p = prior.reindex(teams).fillna(0.0).values
    d = np.linalg.solve(X.T @ X + LAM * np.eye(len(teams)), X.T @ (y - X @ p))
    out = prior.reindex(prior.index.union(teams)).fillna(0.0)
    out.loc[teams] = p + d
    return out


def build_priors(year, fbs_now):
    """Every variant's preseason prior for `year`, plus the measured FCS gap."""
    prev = load_games(year - 1)
    sp_raw = M.load_json(f"sp_{year - 1}.json")
    sp = pd.Series({r["team"]: r["rating"] for r in sp_raw if r.get("rating") is not None})
    prev_div = divisions(prev)
    sp = sp[[t for t in sp.index if prev_div.get(t) == "fbs"]]
    sp -= sp.mean()

    def blend(own):
        teams = set(fbs_now) | set(sp.index) | set(own.index)
        s, o = sp.reindex(teams), own.reindex(teams).fillna(0.0)
        #No prior-season SP+ (a team that just moved up): use its own rating alone
        #rather than blending with 0. Production gets a real preseason SP+ for these
        #teams, which the leak-free backtest can't, so this stands in for it.
        return (SP_WEIGHT * s + (1 - SP_WEIGHT) * o).fillna(o)

    # --- production: last season fit on games involving an FBS team ---
    own_prod = massey(prev[involves_fbs(prev)])
    prod = blend(own_prod)

    # --- unified: last season fit on every FBS/FCS game, FBS mean pinned to 0 ---
    own_all = massey(prev[fbs_or_fcs_only(prev)])
    prev_fbs = [t for t, c in prev_div.items() if c == "fbs" and t in own_all.index]
    prev_fcs = [t for t, c in prev_div.items() if c == "fcs" and t in own_all.index]
    own_all -= own_all[prev_fbs].mean()
    fcs_mean = float(own_all[prev_fcs].mean())
    uni = blend(own_all)
    uni -= uni.reindex(fbs_now).dropna().mean()

    return prod, own_all, uni, fcs_mean


def seed_fcs(prior, fcs_teams, fbs_now, value):
    """Every FCS team starts at `value` relative to the average FBS prior."""
    out = prior.copy()
    base = out.reindex(fbs_now).dropna().mean()
    out = out.reindex(out.index.union(fcs_teams))
    out.loc[list(fcs_teams)] = base + value
    return out


def unified_fcs(uni, own_all, fcs_teams, fcs_mean):
    """FCS team prior = half its own last-season rating, half the FCS average;
    a team with no FCS history last season starts at the FCS average."""
    out = uni.reindex(uni.index.union(fcs_teams))
    for t in fcs_teams:
        own = own_all.get(t, np.nan)
        out[t] = fcs_mean if pd.isna(own) else SP_WEIGHT * fcs_mean + (1 - SP_WEIGHT) * own
    return out


def run():
    rows, gaps = [], []
    for year in SEASONS:
        df = load_games(year)
        div = divisions(df)
        fbs_now = sorted(t for t, c in div.items() if c == "fbs")
        fcs_now = sorted(t for t, c in div.items() if c == "fcs")
        lines = M.build_line_lookup(M.load_json(f"lines_{year}.json"))

        prod, own_all, uni, fcs_mean = build_priors(year, fbs_now)
        gaps.append({"season": year, "fcs_minus_fbs_last_season": round(fcs_mean, 1)})

        priors = {"prod": prod, "unified": unified_fcs(uni, own_all, fcs_now, fcs_mean)}
        for c in FIXED_OFFSETS:
            priors[f"fixed_{c}"] = seed_fcs(prod, fcs_now, fbs_now, c)

        fbs_games = df[involves_fbs(df)]
        all_games = df[fbs_or_fcs_only(df) | involves_fbs(df)]
        variants = {
            "prod": ("prod", fbs_games),
            "unified": ("unified", fbs_games),
            "unified_all": ("unified", all_games),
            **{f"fixed_{c}": (f"fixed_{c}", fbs_games) for c in FIXED_OFFSETS},
            **{f"fixed_{c}_all": (f"fixed_{c}", all_games) for c in FIXED_OFFSETS},
        }

        for W in range(1, int(df["week"].max()) + 1):
            test = df[df["week"] == W]
            if len(test) == 0:
                continue
            fits = {name: ridge_to_prior(games[games["week"] < W], priors[pkey])
                    for name, (pkey, games) in variants.items()}

            #How many FCS opponents each team had played before this week -- the
            #quantity a "boost" would scale with.
            past = df[df["week"] < W]
            fcs_played = {}
            for _, g in past.iterrows():
                if g["hc"] == "fbs" and g["ac"] == "fcs":
                    fcs_played[g["homeTeam"]] = fcs_played.get(g["homeTeam"], 0) + 1
                if g["ac"] == "fbs" and g["hc"] == "fcs":
                    fcs_played[g["awayTeam"]] = fcs_played.get(g["awayTeam"], 0) + 1

            for _, g in test.iterrows():
                kind = "fbs_fbs" if (g["hc"] == "fbs" and g["ac"] == "fbs") else (
                    "fbs_fcs" if {g["hc"], g["ac"]} == {"fbs", "fcs"} else None)
                if kind is None:
                    continue
                h, a, n = g["homeTeam"], g["awayTeam"], bool(g["neutralSite"])
                rec = {
                    "season": year, "week": W, "kind": kind, "home": h, "away": a,
                    "actual": float(g["homePoints"] - g["awayPoints"]),
                    "spread": lines.get((h, a)),
                    "fcs_diff": fcs_played.get(h, 0) - fcs_played.get(a, 0),
                }
                for name, r in fits.items():
                    rec[name] = float(r.get(h, 0.0) - r.get(a, 0.0) + (0.0 if n else HFA))
                rows.append(rec)
        print(f"  {year}: FCS average sat {fcs_mean:+.1f} vs the FBS average last season")

    tbl = pd.DataFrame(rows)
    names = [c for c in tbl.columns if c == "prod" or c.startswith(("fixed_", "unified"))]
    tbl.to_csv(os.path.join(RESULTS_DIR, "fcs_prior_detail.csv"), index=False)
    pd.DataFrame(gaps).to_csv(os.path.join(RESULTS_DIR, "fcs_gap_by_season.csv"), index=False)
    return tbl, names


def summarise(tbl, names, kind, weeks_label, week_mask):
    sub = tbl[(tbl["kind"] == kind) & week_mask]
    base_err = (sub["prod"] - sub["actual"]).abs()
    out = []
    for name in names:
        err = (sub[name] - sub["actual"]).abs()
        diff = err - base_err
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        signed = sub[name] - sub["actual"]
        #Boost check: regress signed error on (home FCS games played - away's). A
        #positive slope means each FCS game played inflates a team's prediction.
        x = sub["fcs_diff"].values
        slope = np.polyfit(x, signed.values, 1)[0] if x.std() > 0 else np.nan
        wl = sub.dropna(subset=["spread"])
        vegas = -wl["spread"]
        big = wl[vegas.abs() >= 10]
        flips = int((np.sign(big[name]) != np.sign(-big["spread"])).sum())
        su = [grade_pick(p, None, act, act > 0)["straight_up_correct"] for p, act in zip(sub[name], sub["actual"])]
        out.append({
            "window": weeks_label, "games": kind, "variant": name, "n": len(sub),
            "MAE": round(err.mean(), 3),
            "MAE_vs_prod": round(diff.mean(), 3),
            "se": round(se, 3),
            "SU_%": round(100 * np.mean(su), 1),
            "vs_line_MAE": round((wl[name] - vegas).abs().mean(), 2),
            "flips_vs_10pt_fav": flips,
            "boost_pts_per_fcs_game": round(slope, 2),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    print("Replaying 2022-2025...")
    tbl, names = run()
    early = tbl["week"].isin(EARLY_WEEKS)
    res = pd.concat([
        summarise(tbl, names, "fbs_fbs", "weeks 1-4", early),
        summarise(tbl, names, "fbs_fbs", "weeks 5+", ~early),
        summarise(tbl, names, "fbs_fcs", "all weeks", early | ~early),
    ], ignore_index=True)
    res.to_csv(os.path.join(RESULTS_DIR, "fcs_prior_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    for (w, k), g in res.groupby(["window", "games"], sort=False):
        print(f"\n=== {k.upper()} · {w} ===")
        print(g.drop(columns=["window", "games"]).to_string(index=False))
