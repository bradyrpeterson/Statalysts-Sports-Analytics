#Import needed packages
import pandas as pd
import numpy as np
import cfbd #College Football Data API
import requests #For pulling data from CFBD
import json #For handling json files
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
#Using configuration suggested by CFBD turn the games into a dataset
configuration = cfbd.Configuration(
    access_token = api_key)
headers = {"Authorization": f"Bearer {api_key}"}

#Only care about games where one of the teams was FBS
# Load list of FBS teams (static roster file, doesn't need refreshing)
with open("football/fbs_teams_2026.json", "r") as f:
    fbs_teams = json.load(f)

need_cols = ["season","seasonType","week","startDate","startTimeTBD","venue","venueId",
             "homeTeam","awayTeam","homePoints","awayPoints","homeConference","awayConference","neutralSite"]

#--- Preseason prior --------------------------------------------------------
#A 50/50 blend of SP+ and our own rating regression on last season's results.
#Measured over 2022-2025 weeks 1-4 (research/model_backtest/early_backtest.py):
#the blend gave MAE 13.88 vs 14.25 for SP+ alone and 13.93 for last-year alone.
#The margin over last-year-alone is slim, and that test used the PRIOR season's
#final SP+ rather than a true preseason projection (using same-season SP+ in a
#backtest would leak the season being predicted), so it understates what SP+
#contributes here -- live, CFBD serves an actual preseason projection.
SP_PLUS_WEIGHT = 0.5
OWN_MODEL_WEIGHT = 0.5

#--- How this season's results move a team off its prior --------------------
#Ratings are a Massey point-margin regression fit with ridge shrinkage toward
#the preseason prior above, i.e. minimising ||y - Xb||^2 + LAMBDA*||b - prior||^2.
#With no games played this returns the prior exactly; as results accumulate it
#converges on the ordinary Massey fit.
#
#This replaces a global prior/in-season blend weight. The distinction matters:
#a blend weight extends the same trust to every team's record regardless of
#whether that team's schedule can actually support an estimate yet, whereas
#ridge shrinkage is per-team and proportional to that team's own evidence.
#Early in the season there are far more teams than games, so the unshrunk
#regression is underdetermined and returns arbitrary values -- that is what
#produced ratings like a 25-point home-field advantage, and predictions on the
#wrong side of games the market had as three-score blowouts.
#
#LAMBDA was swept over 2022-2025 (research/model_backtest/sweep_lambda.py).
#Values of 2-4 were jointly optimal in weeks 1-4 AND weeks 5+ -- there was no
#early/late tradeoff to split. It also agrees with the value theory suggests,
#(margin noise variance)/(prior error variance) ~= 13^2/7^2 ~= 3.5. Caveat: the
#sweep was tuned and evaluated on the same four seasons.
RIDGE_TO_PRIOR_LAMBDA = 3.0

#Home field is pinned rather than re-fit. On small samples it is not separately
#identifiable from team strength, and re-fitting it weekly produced values as
#absurd as 25 points. 2.5 is the long-run college-football figure.
HOME_FIELD_ADVANTAGE = 2.5

#Weeks of this season's results needed before betting edges are published.
#Predictions themselves are shown from week 1 -- straight-up accuracy in weeks
#1-4 measured 74.7%, actually higher than weeks 5+ (70.6%), because early
#schedules carry more mismatches. Against the spread is the opposite story:
#weeks 1-4 graded 47.4% ATS versus 50.1% later, so early edges are not worth
#publishing even though the predictions are.
MODEL_FULLY_TRAINED_MIN_WEEKS = 4

#Don't hit the CFBD API more than once per this many seconds -- refresh() is
#called at the top of every page load (see app.py) so the site never shows
#stale scores without a redeploy, but a burst of page views (or one visitor
#clicking between pages) shouldn't turn into a burst of redundant API calls.
MIN_REFRESH_INTERVAL_SECONDS = 3600
_last_refresh_time = 0.0


def _fit_team_ratings(games_df):
    """Home/away indicator regression on margin -- same method used for both
    the live in-season ratings below and the backtest that picked the preseason
    blend (see PRESEASON PRIOR section). Returns (ratings Series, home_field)."""
    if len(games_df) == 0:
        return pd.Series(dtype=float), 0.0

    #Build the design matrix
    teams_in = sorted(set(games_df["homeTeam"]).union(games_df["awayTeam"]))
    #Home teams get a +1 value and -1 is for away
    #This setup allows for the regression to assign each team a numeric rating
    X = pd.DataFrame(0, index=np.arange(len(games_df)), columns=teams_in)
    for i, row in games_df.reset_index(drop=True).iterrows():
        X.loc[i, row["homeTeam"]] = 1    # +1 for home team
        X.loc[i, row["awayTeam"]] = -1   # -1 for away team

    #Add home field column
    X["home_field"] = 1

    #Time to fit the margin
    #The regressions finds coefficients that best fit the margins
    #Don't worry about an intercept cause we have home field
    y = games_df["margin"].reset_index(drop=True)

    #Create the linear regression based on the margins
    #Essentially finds a set of ratings that makes the predicted scores as close
    #as possible to what actually happened
    model = LinearRegression(fit_intercept=False)
    #Fit a model that predicts y (point margin) as a linear
    #combination of the columns in X
    model.fit(X, y)

    #Ensure home field is counted for in the team ratings
    #Every team gets a numeric rating
    #home_field isolates the average value that being at home bears
    #Substrating the mean centers the ratings at 0 so the average team has that rating
    coefs = pd.Series(model.coef_, index=X.columns)
    hf = coefs["home_field"]
    r = coefs.drop("home_field")
    #Make the average team=0
    r -= r.mean()
    return r, hf


def _fit_ridge_to_prior(games_df, prior, lam=RIDGE_TO_PRIOR_LAMBDA):
    """Massey point-margin regression shrunk toward `prior` rather than toward zero.

    Minimises ||y - Xb||^2 + lam*||b - prior||^2, so a team only moves off its
    preseason rating to the extent its own results argue for it. With no games
    played this returns the prior unchanged; with a full season it converges on
    the ordinary Massey fit. Home field is held fixed rather than fit, so `y` is
    the observed margin with home field already subtracted out.
    """
    if len(games_df) == 0:
        return prior.copy()

    teams = sorted(set(games_df["homeTeam"]).union(games_df["awayTeam"]))
    index = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(games_df), len(teams)))
    y = np.zeros(len(games_df))
    for i, (_, row) in enumerate(games_df.iterrows()):
        X[i, index[row["homeTeam"]]] = 1.0
        X[i, index[row["awayTeam"]]] = -1.0
        hfa = 0.0 if row.get("neutralSite", False) else HOME_FIELD_ADVANTAGE
        y[i] = (row["homePoints"] - row["awayPoints"]) - hfa

    #Non-FBS opponents appear on schedules but have no prior entry -- start them
    #at 0 (an average team) and let their results move them from there.
    p = prior.reindex(teams).fillna(0.0).values
    delta = np.linalg.solve(X.T @ X + lam * np.eye(len(teams)), X.T @ (y - X @ p))

    ratings = prior.reindex(prior.index.union(teams)).fillna(0.0)
    ratings.loc[teams] = p + delta
    return ratings


def load_data():
    """Fetch everything fresh from CFBD (games, venues, SP+, last season) and refit
    the ratings on it. This is the entire body of what used to run once at import
    time -- now it's callable, so refresh() can re-run it on every page load
    instead of only on process start."""
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        games = api_instance.get_games(year=2026)

        venues_api = cfbd.VenuesApi(api_client)
        try:
            venues = venues_api.get_venues()
        except Exception as e:
            print(f"Error fetching venues: {e}")
            venues = []

        #SP+ (Bill Connelly's advanced rating) for the current season -- this early,
        #before many/any games are played, it *is* the preseason projection. Used
        #below as half of the preseason prior for team ratings.
        try:
            sp_plus = cfbd.RatingsApi(api_client).get_sp(year=2026)
        except Exception as e:
            print(f"Error fetching SP+ ratings: {e}")
            sp_plus = []

        #Last season's own completed games -- the other half of the preseason
        #prior (see PRESEASON PRIOR at the top of this file).
        try:
            last_season_games = cfbd.GamesApi(api_client).get_games(year=2025)
            last_season_fbs = [t.school for t in cfbd.TeamsApi(api_client).get_fbs_teams(year=2025)]
        except Exception as e:
            print(f"Error fetching last season's games: {e}")
            last_season_games, last_season_fbs = [], []

    #Map venue id -> "City, State" so game locations can be shown alongside the venue name
    venue_location = {}
    for v in venues:
        v = v.to_dict()
        city_state = ", ".join(p for p in [v.get("city"), v.get("state")] if p)
        if v.get("id") is not None and city_state:
            venue_location[v["id"]] = city_state

    #Each game is turned into a dictionary then into a dataframe
    df = pd.DataFrame([g.to_dict() for g in games])
    #Only keep the columns that matter
    df = df[need_cols].copy()
    #Only keep games if it involved an FBS team
    df = df[df["homeTeam"].isin(fbs_teams) | df["awayTeam"].isin(fbs_teams)]
    df = df.reset_index(drop=True)

    #Need to make an upcoming data frame as well as a completed data frame
    completed = df.dropna(subset=["homePoints", "awayPoints"]).reset_index(drop=True)
    upcoming = df[df["homePoints"].isna() | df["awayPoints"].isna()].reset_index(drop=True)
    regular_upcoming = upcoming[upcoming["seasonType"] == "regular"]

    if len(regular_upcoming) > 0:
        next_week = int(regular_upcoming["week"].dropna().sort_values().unique()[0])
    else:
        next_week = 17

    #define what margin is
    #sort the dataframe to have a line of home and away teams
    completed = completed.copy()
    completed["margin"] = completed["homePoints"] - completed["awayPoints"]

    #Last season's own final ratings -- the other half of the preseason prior.
    last_season_df = pd.DataFrame([g.to_dict() for g in last_season_games])
    if len(last_season_df) > 0:
        last_season_df = last_season_df[
            last_season_df["homeTeam"].isin(last_season_fbs) | last_season_df["awayTeam"].isin(last_season_fbs)
        ]
        last_season_df = last_season_df.dropna(subset=["homePoints", "awayPoints"]).reset_index(drop=True)
        last_season_df["margin"] = last_season_df["homePoints"] - last_season_df["awayPoints"]
    own_last_year_rating, _ = _fit_team_ratings(last_season_df)

    preseason_rating_sp = pd.Series(
        {t.team: t.rating for t in sp_plus if t.team in fbs_teams}
    )
    if len(preseason_rating_sp) > 0:
        preseason_rating_sp -= preseason_rating_sp.mean()

    all_prior_teams = set(fbs_teams) | set(preseason_rating_sp.index) | set(own_last_year_rating.index)
    sp_full = preseason_rating_sp.reindex(all_prior_teams).fillna(0.0)
    own_full = own_last_year_rating.reindex(all_prior_teams).fillna(0.0)
    preseason_rating = SP_PLUS_WEIGHT * sp_full + OWN_MODEL_WEIGHT * own_full

    regular_completed_weeks = completed[completed["seasonType"] == "regular"]["week"].dropna()
    weeks_completed = int(regular_completed_weeks.max()) if len(regular_completed_weeks) > 0 else 0

    #Shrink toward the preseason prior in proportion to how much each team's own
    #results actually argue for moving. Before any games are played this is
    #exactly the prior.
    ratings = _fit_ridge_to_prior(completed, preseason_rating)
    home_field = HOME_FIELD_ADVANTAGE

    # Create FBS-only rankings
    FBS_rankings = pd.DataFrame({'team': ratings.index, 'rating': ratings.values})
    FBS_rankings = FBS_rankings[FBS_rankings['team'].isin(fbs_teams)]
    FBS_rankings = FBS_rankings.sort_values(by='rating', ascending=False).reset_index(drop=True)

    #Gates whether betting edges are published (see MODEL_FULLY_TRAINED_MIN_WEEKS);
    #predictions themselves are shown from week 1.
    model_fully_trained = weeks_completed >= MODEL_FULLY_TRAINED_MIN_WEEKS

    return {
        "games": games,
        "venue_location": venue_location,
        "completed": completed,
        "upcoming": upcoming,
        "next_week": next_week,
        "ratings": ratings,
        #Exposed so a rating set can be refit as of an earlier point in the season
        #(see ratings_as_of) -- the prior is the fixed half of that calculation.
        "preseason_rating": preseason_rating,
        "home_field": home_field,
        "FBS_rankings": FBS_rankings,
        "weeks_completed": weeks_completed,
        "model_fully_trained": model_fully_trained,
    }


def refresh(force=False):
    """Re-fetch everything from CFBD and retrain the model, replacing this
    module's data in place. Call this before serving any page (see app.py)
    so the site always reflects the current games/scores instead of whatever
    was live when the process last started -- no redeploy required.

    Throttled to once every MIN_REFRESH_INTERVAL_SECONDS so back-to-back page
    loads (or several visitors at once) don't turn into a burst of redundant
    CFBD calls; pass force=True to bypass that (e.g. an admin refresh button)."""
    global _last_refresh_time
    now = time.monotonic()
    if not force and (now - _last_refresh_time) < MIN_REFRESH_INTERVAL_SECONDS:
        return
    globals().update(load_data())
    _last_refresh_time = now


# Load data once at import so the module works even if nobody calls refresh()
# (e.g. a script importing this directly). app.py calls refresh() again at
# the top of every request.
refresh(force=True)


def ratings_as_of(week):
    """The rating set the model would have held going into `week`.

    Fits the same ridge-to-prior on only the regular-season games completed before
    that week, so a pick can be graded against the numbers that were actually live
    when it was made rather than against a model that has since seen the result.
    """
    prior_games = completed[
        (completed["seasonType"] == "regular") & (completed["week"].astype(float) < float(week))
    ]
    return _fit_ridge_to_prior(prior_games, preseason_rating)


def predict_game_with(ratings_set, home, away, neutral_site=False):
    """predict_game against a specific rating set (e.g. one from ratings_as_of)."""
    margin = (ratings_set[home] - ratings_set[away]) + (0 if neutral_site else HOME_FIELD_ADVANTAGE)
    prob = 1 / (1 + np.exp(-margin / 7))
    return margin, prob


#Prediciton function
def predict_game(home, away, neutral_site=False):
    """Predicted home margin and home win probability.

    The margin is the rating difference plus home field. Team ratings already
    carry everything the model knows -- an earlier version ran a second
    regression on top of these ratings using box-score stats, but that layer
    measured worse than the ratings alone in every window tested (2022-2025:
    49.6% ATS against 51.7%, and 26 predictions on the wrong side of a
    double-digit market favourite in weeks 1-4 against 1), so it was removed.
    """
    margin = (ratings[home] - ratings[away]) + (0 if neutral_site else home_field)
    #Rough logistic: a team favoured by 7 wins about 75% of the time. This is a
    #stated assumption, not a fitted calibration -- see the research directory.
    prob = 1 / (1 + np.exp(-margin / 7))
    return margin, prob

def get_betting_lines(week, year=2026, season_type="regular"):
   #Get draftkings specific betting lines for the week
    if season_type == "postseason":
        # Postseason doesn't use week numbers
        lines_url = f"https://api.collegefootballdata.com/lines?year={year}&seasonType=postseason"
    else:
        lines_url = f"https://api.collegefootballdata.com/lines?year={year}&week={week}&seasonType=regular"

    try:
        response = requests.get(lines_url, headers=headers)
        response.raise_for_status()
        lines_data = response.json()

        betting_lines = {}

        for game in lines_data:
            home = game.get("homeTeam")
            away = game.get("awayTeam")
            lines = game.get("lines", [])

            if not home or not away or not lines:
                continue

            # Look specifically for DraftKings line
            draftkings_spread = None
            for line in lines:
                if line.get("provider") == "DraftKings":
                    draftkings_spread = line.get("spread")
                    break

            if draftkings_spread is not None:
                betting_lines[(home, away)] = draftkings_spread

        return betting_lines

    except Exception as e:
        print(f"Error fetching betting lines: {e}")
        return {}

def calculate_edge_highlight(model_margin, betting_spread):
    # Convert betting spread to match our model's convention
    # If betting spread is +7, that means home is favored by 7
    # If betting spread is -7, that means away is favored by 7
    # We need to flip it to match our model's convention (positive = home favored)
    betting_margin = -betting_spread

    # Calculate the difference
    difference = abs(model_margin - betting_margin)

    if difference >= 5:
        return 'edge-big'
    elif difference >= 3:
        return 'edge-medium'
    else:
        return None

def get_upcoming_predictions(week=None,conference=None):
    # Use the upcoming games dataset (no scores yet)
    games_to_predict = upcoming.copy()

    if week is not None:
        if int(week)>16:
            games_to_predict = games_to_predict[games_to_predict["seasonType"]== "postseason"]
        else:
            games_to_predict = games_to_predict[(games_to_predict["week"].astype(int) == int(week))&(games_to_predict["seasonType"]=="regular")]
    else:
        games_to_predict =games_to_predict[games_to_predict["seasonType"]=="regular"]
    #Filter by conferences
    if conference is not None:
        games_to_predict = games_to_predict[
            (games_to_predict["homeConference"] == conference) |
            (games_to_predict["awayConference"] == conference)
        ]
    # Fetch betting lines for this week
    if week and int(week) > 16:
        betting_lines = get_betting_lines(week, season_type="postseason")
    else:
        betting_lines = get_betting_lines(week if week else next_week, season_type="regular")

    #This season's win-loss record per team, shown alongside each matchup.
    records = {}
    for _, g in completed.iterrows():
        home_won = g["homePoints"] > g["awayPoints"]
        for team, won in ((g["homeTeam"], home_won), (g["awayTeam"], not home_won)):
            wins, losses = records.get(team, (0, 0))
            records[team] = (wins + 1, losses) if won else (wins, losses + 1)

    predictions = []
    for _, game in games_to_predict.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]

        is_neutral = game.get("neutralSite",False)

        #Skip games missing a team rating, and require both teams to be FBS: a
        #non-FBS opponent (FCS "buy games" etc.) does get a rating out of the
        #regression since it played an FBS team, but that number means nothing --
        #it's fit on a handful of lopsided games and centred against FBS
        #competition, not against its own level. That's what produced spreads
        #20-35 points off Vegas for exactly these matchups.
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in fbs_teams or away not in fbs_teams:
            continue

        try:
            margin, prob = predict_game(home, away, neutral_site=is_neutral)
            winner = home if margin > 0 else away

            # Get betting line for this game
            betting_spread = betting_lines.get((home, away), None)

            # Calculate edge highlight
            edge_class = None
            spread_diff = None
            if betting_spread is not None:
                edge_class = calculate_edge_highlight(margin, betting_spread)
                # Calculate the actual difference for display
                betting_margin = -betting_spread
                spread_diff = round(margin - betting_margin, 1)

            # Before MODEL_FULLY_TRAINED_MIN_WEEKS the edge is published but flagged
            # provisional: weeks 1-4 backtested at 47.4% against the spread versus
            # 50.1% later, so these are worth showing but not worth putting in the
            # headline record. tracking.py carries the flag through and leaves
            # provisional picks out of the profit/ROI numbers while still displaying
            # them. An earlier version cleared edge_class outright, which hid weeks
            # 1-4 from the results panel entirely.
            provisional = not model_fully_trained

            game_date = pd.to_datetime(game.get("startDate"), utc=True, errors="coerce")
            game_date_et = game_date.tz_convert("America/New_York") if pd.notna(game_date) else None
            is_tbd = bool(game.get("startTimeTBD", False))

            venue_name = game.get("venue")
            city_state = venue_location.get(game.get("venueId"))
            if venue_name and city_state:
                location = f"{venue_name} — {city_state}"
            else:
                location = venue_name or city_state

            predictions.append({
                "home": home,
                "away": away,
                "home_record": "%d-%d" % records.get(home, (0, 0)),
                "away_record": "%d-%d" % records.get(away, (0, 0)),
                "predicted_winner": winner,
                "margin": round(abs(margin), 2),
                "prob": round(prob * 100, 1) if margin > 0 else round((1 - prob) * 100, 1),
                "betting_spread": betting_spread,
                "edge_class": edge_class,
                "spread_diff": spread_diff,
                "provisional": provisional,
                #Carried into tracked_picks so the results panel can show a whole
                #week's slate rather than guessing at one from the dates.
                "week": int(game["week"]) if pd.notna(game.get("week")) else None,
                #CFBD hands back an aenum here. It subclasses str, so it compares and
                #stores like one, but str() on it renders "SeasonType.REGULAR" -- take
                #the plain value so what lands in Firestore is an ordinary string.
                "season_type": getattr(game.get("seasonType"), "value", game.get("seasonType")),
                "neutral_site": is_neutral,
                "date": game_date.strftime("%Y-%m-%d") if pd.notna(game_date) else None,
                "game_date_display": game_date_et.strftime("%a, %b %-d") if game_date_et is not None else None,
                "game_time_display": ("TBD" if is_tbd else game_date_et.strftime("%-I:%M %p ET")) if game_date_et is not None else None,
                "kickoff_iso": game_date_et.isoformat() if game_date_et is not None else None,
                "location": location,
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    return pd.DataFrame(predictions)



# How to print if I wasn't using the app
#m, p = predict_game("Florida State", "Ohio State")
#print(f"\nPredicted margin {m:.2f}, win probability {p*100:.1f}%")
