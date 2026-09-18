# Statalysts

A college football and basketball prediction engine, built on transparent power ratings and validated with a walk-forward backtest.

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-46E3B7?logo=render&logoColor=white)](https://statalysts.com)

**Live at [statalysts.com](https://statalysts.com)** — free, no payment required.

---

## What this is

Every FBS team carries a numeric power rating derived from point margins across the season. A game prediction is the difference between two ratings, plus home field. That's the whole model — no black box, and every number on the site traces back to something you can recompute from this repo.

The interesting engineering problem isn't the rating itself; it's the early season. Through week 3 there are more teams than games played, so a plain ratings regression is underdetermined and returns arbitrary values. This project's answer is to shrink toward a preseason prior in proportion to how much each team's own results actually support moving.

**Football** — power ratings, weekly predictions for every FBS matchup, DraftKings line comparison, and a graded record of past picks.
**Basketball** — daily D1 predictions and ratings. Still on the older two-stage model, and flagged Beta on the site accordingly.

---

## How the model works

Team ratings solve a ridge-regularized least squares problem:

```
ratings = argmin ‖y − Xb‖² + λ‖b − prior‖²
```

- `X` — one row per completed game: `+1` for the home team, `−1` for the away team
- `y` — the final margin, with home field already subtracted out
- `prior` — a 50/50 blend of SP+'s preseason projection and a rating fit on last season's results. FCS teams get the same blend with the FCS average standing in for SP+, on the same scale — nearly half of September's games are FBS-vs-FCS, so an FCS team seeded like an average FBS side inflated every FBS team that beat one
- `λ = 3` — how hard this season's results must argue to move a team off its prior

A prediction is then just:

```python
margin = ratings[home] - ratings[away] + home_field   # home_field = 2.5, or 0 at a neutral site
prob   = 1 / (1 + exp(-margin / 7))
```

**Why shrink toward a prior instead of blending?** A global blend weight extends the same trust to every team's record regardless of whether that team's schedule can support an estimate yet. Ridge shrinkage is per-team and proportional to each team's own evidence, so teams in sparsely-connected corners of the schedule graph stay near their prior instead of absorbing the same dose of an underdetermined fit as everyone else.

`λ` was selected by sweeping it across four seasons ([`sweep_lambda.py`](research/model_backtest/sweep_lambda.py)). Values of 2–4 were jointly optimal early *and* late in the season — there was no tradeoff to split — and that range agrees with what theory suggests, roughly (margin noise variance) / (prior error variance) ≈ 13² / 7² ≈ 3.5.

Home field is pinned at 2.5 points rather than re-fit. On small samples it isn't separately identifiable from team strength, and re-fitting it weekly produced values as absurd as 25 points.

---

## How it's validated

Model changes here are decided by backtest, not by intuition. The harness in [`research/model_backtest/`](research/model_backtest) replays four completed seasons (2022–2025) week by week and grades the output exactly the way the live site grades picks.

**Leakage controls.** Every model predicting week *W* is fit using only games from weeks `< W` of that season. Team stats are truncated to what had been published before that week. SP+ priors are always the *prior* season's, never the season under test — same-season SP+ would embed the very results being predicted. Elo ratings are computed strictly sequentially.

**What gets measured.**

- *Mean absolute error* on predicted margin — the metric with the most signal, and the one used to choose between models
- *Straight-up accuracy* — did the pick win outright
- *Against the spread* — graded at −110 with a flat stake, on the same edge threshold the site flags
- *Tail behavior* — how often the model lands on the wrong side of a game the market considers a blowout. A model can have respectable average error and still be unshippable if it does this, so it gates any change reaching production.

The harness also compares alternatives on identical data — Massey with capped margins, Colley, Elo, Bayesian ridge ratings, gradient boosting, and a bounded ML residual layer — so a change has to beat the field rather than just look reasonable. This is how the previous two-stage model (a second regression stacked on the ratings, fed box-score stats) was found to measure *worse than the ratings alone*, and removed.

> **Current figures aren't published here yet.** The rating engine was rewritten recently, and the numbers on record describe the model it replaced. Results will go back in once the current version has a real track record behind it.

---

## Project structure

```
├── app.py                      # Flask routes, auth, Stripe (dormant), scheduled jobs
├── tracking.py                 # Snapshot picks pre-kickoff, grade them, aggregate the record
├── football/predictor.py       # Ratings, predictions, betting lines
├── basketball/predictor.py     # Basketball equivalent (older two-stage model)
├── templates/                  # Jinja templates
├── static/style.css            # Single stylesheet, dark theme
├── scripts/                    # One-off data import utilities
└── research/model_backtest/    # Walk-forward backtest — the evidence behind the model
    ├── backtest.py             # Massey / Colley / Elo / GBR / ensemble comparison
    ├── early_backtest.py       # Weeks 1–4, the regime the original backtest never covered
    ├── sweep_lambda.py         # Tunes the shrinkage parameter
    ├── fcs_prior_backtest.py   # Where FCS teams should start (they were starting far too high)
    ├── residual_backtest.py    # Tests a bounded ML residual layer on top of the ratings
    └── results/                # Summary CSVs (per-game detail is gitignored)
```

---

## Tech stack

**Backend** Flask 3.1 · Python 3.12 · gunicorn
**Modeling** scikit-learn · NumPy · pandas
**Data** Firebase Firestore (auth, pick tracking) · APScheduler (nightly grading)
**Deployment** Render

---

## Running locally

```bash
git clone https://github.com/bradyrpeterson/Combined-Sports-Analytics.git
cd Combined-Sports-Analytics

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# API key from collegefootballdata.com/key
echo "API_KEY=your_cfbd_api_key" > .env

python app.py
```

Then open <http://localhost:5000>.

Firestore powers login and pick tracking. Without `firebase_credentials.json` the app still starts and predictions still render, but authenticated pages and the results panel won't populate.

### Reproducing the backtest

```bash
cd research/model_backtest
python fetch_data.py     # caches ~76MB of CFBD responses (gitignored, one-time)
python fetch_early.py    # weeks 1-3 stats + SP+ ratings
python backtest.py       # model comparison
python early_backtest.py # early-season regime
python sweep_lambda.py   # shrinkage parameter sweep
```

Every fit is walk-forward and uses only data available before the week being predicted. SP+ priors are always the *prior* season's, never the season under test.

---

## Data sources

- [CollegeFootballData API](https://collegefootballdata.com) — games, box scores, SP+ ratings, betting lines
- [CollegeBasketballData API](https://api.collegebasketballdata.com) — basketball games and stats
- Betting lines are DraftKings where available, served through CFBD

---

## Disclaimer

For entertainment and educational purposes only. This is not betting advice, and no edge against the betting market is claimed.

---

## Author

**Brady Peterson**
[GitHub](https://github.com/bradyrpeterson) · [LinkedIn](https://www.linkedin.com/in/brady-peterson-b5ab02308/)
