"""
Model implementations for the backtest. Every "fit_*" function takes only data that
would have been available before the prediction week (the caller in backtest.py is
responsible for slicing train/stats/core data walk-forward -- these functions never
reach outside what they're handed).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(HERE, "data")


def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def load_fbs_teams():
    with open(os.path.join(REPO_ROOT, "football", "fbs_teams_2025.json")) as f:
        return set(json.load(f))


def load_games_df(year):
    raw = load_json(f"games_{year}.json")
    df = pd.DataFrame(raw)
    need_cols = [
        "season", "seasonType", "week", "startDate", "homeTeam", "awayTeam",
        "homePoints", "awayPoints", "homeConference", "awayConference",
        "neutralSite", "completed",
    ]
    df = df[need_cols].copy()
    df["neutralSite"] = df["neutralSite"].fillna(False)
    return df


def build_stats_clean(year, start_week, end_week):
    raw = load_json(f"stats_{year}_{start_week}_{end_week}.json")
    df = pd.DataFrame(raw)
    wide = df.pivot_table(index="team", columns="statName", values="statValue", aggfunc="first").reset_index()
    wide["yardsPerPlay_off"] = wide["totalYards"] / (wide["rushingAttempts"] + wide["passAttempts"])
    wide["yardsPerPlay_def"] = wide["totalYardsOpponent"] / (wide["rushingAttemptsOpponent"] + wide["passAttemptsOpponent"])
    wide["thirdDownPct"] = wide["thirdDownConversions"] / wide["thirdDowns"]
    wide["turnoverMargin"] = wide["turnoversOpponent"] - wide["turnovers"]
    useful = ["team", "yardsPerPlay_off", "yardsPerPlay_def", "thirdDownPct", "turnoverMargin"]
    return wide[useful]


def load_core_prior(year_minus_1):
    """Prior-season CORE ratings. Returns None if not cached (e.g. no 2021 file)."""
    path = os.path.join(DATA_DIR, f"core_{year_minus_1}.json")
    if not os.path.exists(path):
        return None
    raw = load_json(f"core_{year_minus_1}.json")
    df = pd.DataFrame(raw)[["team", "overall", "offense", "defense"]]
    return df.set_index("team")


LINE_PROVIDER_PRIORITY = [
    "DraftKings", "consensus", "Bovada", "William Hill (New Jersey)",
    "Caesars Sportsbook (Colorado)", "teamrankings",
]


def build_line_lookup(lines_raw, provider_counter=None):
    """(home, away) -> spread, preferring DraftKings, falling back down the priority
    list (documented deviation: 2022 has zero DraftKings entries in CFBD's data, so we
    substitute 'consensus' for that season; 2023-2025 mostly have DraftKings)."""
    lookup = {}
    for game in lines_raw:
        home, away = game.get("homeTeam"), game.get("awayTeam")
        lines = game.get("lines", [])
        if not home or not away or not lines:
            continue
        by_provider = {l.get("provider"): l.get("spread") for l in lines if l.get("spread") is not None}
        chosen = None
        chosen_provider = None
        for p in LINE_PROVIDER_PRIORITY:
            if p in by_provider:
                chosen = by_provider[p]
                chosen_provider = p
                break
        if chosen is None and by_provider:
            chosen_provider, chosen = next(iter(by_provider.items()))
        if chosen is not None:
            lookup[(home, away)] = chosen
            if provider_counter is not None:
                provider_counter[chosen_provider] += 1
    return lookup


# ---------------------------------------------------------------------------
# Stage 1: Massey-style least-squares team ratings (this is what the current
# production model calls "ratings" -- point-margin regression on team indicators)
# ---------------------------------------------------------------------------

def _design_matrix(train, teams, cap=None):
    X = pd.DataFrame(0, index=np.arange(len(train)), columns=teams)
    y = np.empty(len(train))
    train = train.reset_index(drop=True)
    for i, row in train.iterrows():
        X.loc[i, row["homeTeam"]] = 1
        X.loc[i, row["awayTeam"]] = -1
        m = row["homePoints"] - row["awayPoints"]
        y[i] = np.clip(m, -cap, cap) if cap else m
    X["home_field"] = 1
    return X, y


def fit_massey_ratings(train, cap=None, estimator=None):
    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"]))
    X, y = _design_matrix(train, teams, cap=cap)
    model = estimator if estimator is not None else LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index=X.columns)
    home_field = coefs["home_field"]
    ratings = coefs.drop("home_field")
    ratings -= ratings.mean()
    return ratings, home_field


def predict_rating_only(ratings, home_field, home, away, neutral):
    if home not in ratings.index or away not in ratings.index:
        return None
    diff = ratings[home] - ratings[away]
    hfa = 0 if neutral else home_field
    return float(diff + hfa)


# ---------------------------------------------------------------------------
# Current production model: stage-1 Massey ratings + stage-2 regression on
# [rating_diff, ypp_diff, third_diff, to_diff, home_court, r_home, r_away,
#  ypp_off_home, ypp_off_away, ypp_def_home, ypp_def_away]
# `estimator_factory` lets us swap LinearRegression -> BayesianRidge -> GBR while
# reusing identical feature construction (this is the "same features, different
# fitting paradigm" comparison).
# ---------------------------------------------------------------------------

def _build_stage2_rows(train, stats_clean, ratings, extra_feature_fn=None):
    features, targets = [], []
    stats_idx = stats_clean.set_index("team")
    for _, game in train.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]
        margin = game["homePoints"] - game["awayPoints"]
        neutral = bool(game.get("neutralSite", False))
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_idx.index or away not in stats_idx.index:
            continue
        h, a = stats_idx.loc[home], stats_idx.loc[away]
        rating_diff = ratings[home] - ratings[away]
        ypp_diff = h["yardsPerPlay_off"] - a["yardsPerPlay_def"]
        third_diff = h["thirdDownPct"] - a["thirdDownPct"]
        to_diff = h["turnoverMargin"] - a["turnoverMargin"]
        hc = 0 if neutral else 1
        vec = [
            rating_diff, ypp_diff, third_diff, to_diff, hc,
            ratings[home], ratings[away],
            h["yardsPerPlay_off"], a["yardsPerPlay_off"],
            h["yardsPerPlay_def"], a["yardsPerPlay_def"],
        ]
        if extra_feature_fn is not None:
            extra = extra_feature_fn(home, away)
            if extra is None:
                continue
            vec = vec + extra
        if any(pd.isna(vec)):
            continue
        features.append(vec)
        targets.append(margin)
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    if len(X) == 0:
        return X, y
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def _stage2_feature_vec(ratings, stats_idx, home, away, neutral, extra_feature_fn=None):
    if home not in ratings.index or away not in ratings.index:
        return None
    if home not in stats_idx.index or away not in stats_idx.index:
        return None
    h, a = stats_idx.loc[home], stats_idx.loc[away]
    rating_diff = ratings[home] - ratings[away]
    ypp_diff = h["yardsPerPlay_off"] - a["yardsPerPlay_def"]
    third_diff = h["thirdDownPct"] - a["thirdDownPct"]
    to_diff = h["turnoverMargin"] - a["turnoverMargin"]
    hc = 0 if neutral else 1
    vec = [
        rating_diff, ypp_diff, third_diff, to_diff, hc,
        ratings[home], ratings[away],
        h["yardsPerPlay_off"], a["yardsPerPlay_off"],
        h["yardsPerPlay_def"], a["yardsPerPlay_def"],
    ]
    if extra_feature_fn is not None:
        extra = extra_feature_fn(home, away)
        if extra is None:
            return None
        vec = vec + extra
    if any(pd.isna(vec)):
        return None
    return vec


def fit_stage2_model(train, stats_clean, ratings, estimator, extra_feature_fn=None):
    X, y = _build_stage2_rows(train, stats_clean, ratings, extra_feature_fn)
    if len(X) < 20:
        return None
    estimator.fit(X, y)
    return {
        "ratings": ratings,
        "stats_idx": stats_clean.set_index("team"),
        "model": estimator,
        "extra_feature_fn": extra_feature_fn,
    }


def predict_stage2(state, home, away, neutral):
    if state is None:
        return None
    vec = _stage2_feature_vec(
        state["ratings"], state["stats_idx"], home, away, neutral, state["extra_feature_fn"]
    )
    if vec is None:
        return None
    return float(state["model"].predict(np.array([vec]))[0])


def core_extra_feature_fn(core_prior):
    """Returns a fn(home, away) -> [overall_diff, offense_diff, defense_diff] using
    STRICTLY prior-season CORE ratings (never same-season -- avoids leakage)."""
    if core_prior is None:
        return None

    def fn(home, away):
        if home not in core_prior.index or away not in core_prior.index:
            return None
        h, a = core_prior.loc[home], core_prior.loc[away]
        return [
            float(h["overall"] - a["overall"]),
            float(h["offense"] - a["offense"]),
            float(h["defense"] - a["defense"]),
        ]

    return fn


# ---------------------------------------------------------------------------
# Colley matrix (win/loss only rating, classic BCS-era method)
# ---------------------------------------------------------------------------

def fit_colley(train):
    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    C = np.zeros((n, n))
    wins = np.zeros(n)
    losses = np.zeros(n)
    games = np.zeros(n)
    for _, g in train.iterrows():
        h, a = idx[g["homeTeam"]], idx[g["awayTeam"]]
        games[h] += 1
        games[a] += 1
        C[h, a] -= 1
        C[a, h] -= 1
        if g["homePoints"] > g["awayPoints"]:
            wins[h] += 1
            losses[a] += 1
        elif g["awayPoints"] > g["homePoints"]:
            wins[a] += 1
            losses[h] += 1
    for i in range(n):
        C[i, i] = 2 + games[i]
    b = 1 + (wins - losses) / 2
    r = np.linalg.solve(C, b)
    ratings = pd.Series(r, index=teams)

    diffs, margins, homes = [], [], []
    for _, g in train.iterrows():
        diffs.append(ratings[g["homeTeam"]] - ratings[g["awayTeam"]])
        margins.append(g["homePoints"] - g["awayPoints"])
        homes.append(0 if bool(g.get("neutralSite", False)) else 1)
    X = np.column_stack([diffs, homes])
    coef, *_ = np.linalg.lstsq(X, np.array(margins), rcond=None)
    return {"ratings": ratings, "coef": coef}


def predict_colley(state, home, away, neutral):
    r = state["ratings"]
    if home not in r.index or away not in r.index:
        return None
    diff = r[home] - r[away]
    a, b = state["coef"]
    return float(a * diff + b * (0 if neutral else 1))


# ---------------------------------------------------------------------------
# Bayesian rating (empirical-Bayes ridge / RidgeCV on the Massey design matrix --
# i.e. MAP estimate of team strength under a Gaussian shrinkage prior, with the
# prior variance selected by cross-validation on the training slice only)
# ---------------------------------------------------------------------------

def fit_bayes_rating(train):
    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"]))
    X, y = _design_matrix(train, teams)
    alphas = np.logspace(-1, 3, 25)
    model = RidgeCV(alphas=alphas, fit_intercept=False)
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index=X.columns)
    home_field = coefs["home_field"]
    ratings = coefs.drop("home_field")
    ratings -= ratings.mean()
    return ratings, home_field


# ---------------------------------------------------------------------------
# Elo (sequential, MOV-adjusted, cross-season mean reversion). Computed once
# across the full chronological timeline; backtest.py looks up pre-week snapshots.
# ---------------------------------------------------------------------------

ELO_START = 1500.0
ELO_HFA = 55.0
ELO_K = 20.0
ELO_REVERT = 0.67  # fraction of rating *kept* across a season boundary


def compute_elo(seasons_games):
    """seasons_games: {year: games_df (all seasonType, all rows)} for years in ascending order.
    Returns (snapshots, game_records_df).
    snapshots[(year, week)] = dict of ratings BEFORE that week's games are played.
    snapshots[(year, 'post')] = ratings after the last regular-season week (used for postseason).
    game_records_df has one row per regular-season game with the elo_diff_raw the two
    teams carried INTO that game (pregame, so later fitting the elo->margin scale never
    uses future information).
    """
    ratings = {}
    snapshots = {}
    records = []
    for year in sorted(seasons_games):
        for t in list(ratings):
            ratings[t] = ELO_START + ELO_REVERT * (ratings[t] - ELO_START)
        df = seasons_games[year]
        reg = df[df["seasonType"] == "regular"].dropna(subset=["homePoints", "awayPoints"])
        weeks = sorted(reg["week"].unique())
        for w in weeks:
            snapshots[(year, w)] = dict(ratings)
            wk = reg[reg["week"] == w]
            for _, g in wk.iterrows():
                h, a = g["homeTeam"], g["awayTeam"]
                rh = ratings.get(h, ELO_START)
                ra = ratings.get(a, ELO_START)
                neutral = bool(g.get("neutralSite", False))
                records.append({
                    "year": year, "week": w, "home": h, "away": a,
                    "elo_diff_raw": rh - ra, "neutral": neutral,
                    "margin": g["homePoints"] - g["awayPoints"],
                })
                hfa = 0 if neutral else ELO_HFA
                exp_h = 1 / (1 + 10 ** (-((rh + hfa - ra)) / 400))
                actual_h = 1.0 if g["homePoints"] > g["awayPoints"] else 0.0
                mov = abs(g["homePoints"] - g["awayPoints"])
                ediff_winner = (rh - ra) if g["homePoints"] > g["awayPoints"] else (ra - rh)
                mov_mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * ediff_winner)
                delta = ELO_K * mov_mult * (actual_h - exp_h)
                ratings[h] = rh + delta
                ratings[a] = ra - delta
        snapshots[(year, "post")] = dict(ratings)
    return snapshots, pd.DataFrame(records)


def elo_scale_factor(game_records_df, year, before_week):
    """before_week: an int (only weeks < before_week from `year` are used) or the
    string 'post' meaning "all regular season games of `year`"."""
    if before_week == "post":
        sub = game_records_df[game_records_df["year"] == year]
    else:
        sub = game_records_df[(game_records_df["year"] == year) & (game_records_df["week"] < before_week)]
    if len(sub) < 20:
        return None
    X = np.column_stack([sub["elo_diff_raw"].values, (~sub["neutral"].values).astype(float)])
    y = sub["margin"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_elo(snapshot, coef, home, away, neutral):
    if coef is None:
        return None
    rh = snapshot.get(home)
    ra = snapshot.get(away)
    if rh is None or ra is None:
        return None
    diff = rh - ra
    a, b = coef
    return float(a * diff + b * (0 if neutral else 1))


# ---------------------------------------------------------------------------
# Gradient boosting on the current model's own stage-2 feature set
# ---------------------------------------------------------------------------

def fit_gbr(train, stats_clean, ratings):
    estimator = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    return fit_stage2_model(train, stats_clean, ratings, estimator)
