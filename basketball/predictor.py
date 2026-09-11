##Import needed packages
import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta
import pytz

load_dotenv()
api_key = os.getenv("API_KEY")

# Headers for API requests
headers = {"Authorization": f"Bearer {api_key}"}

#Load d1 teams (static roster file, doesn't need refreshing)
with open("basketball/d1_teams_2026.json", "r") as f:
    d1_teams = json.load(f)

needed_cols = ["season","status","startDate","homeTeam","awayTeam","homePoints","awayPoints","homeConference","awayConference","neutralSite"]
useful = ["team", "off_eff", "def_eff", "tov_rate"]

#Don't hit the CBBD API more than once per this many seconds -- refresh() is
#called at the top of every page load (see app.py) so the site never shows
#stale ratings without a redeploy, but a burst of page views (or one visitor
#clicking between pages) shouldn't turn into a burst of redundant API calls.
MIN_REFRESH_INTERVAL_SECONDS = 3600
_last_refresh_time = 0.0


def train_prediction_model(completed, ratings, stats_clean):
    """
    Train ML model to learn optimal weights from historical data.
    Called fresh by load_data() every refresh so it always fits the
    latest results.
    """

    features = []
    targets = []

    for _, game in completed.iterrows():
        home = game['homeTeam']
        away = game['awayTeam']
        margin = game['margin']
        neutral = game.get('neutralSite', False)

        # Skip if missing data
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_clean['team'].values or away not in stats_clean['team'].values:
            continue

        # Get features
        h_stats = stats_clean[stats_clean['team'] == home].iloc[0]
        a_stats = stats_clean[stats_clean['team'] == away].iloc[0]

        rating_diff = ratings[home] - ratings[away]
        oeff_diff = h_stats['off_eff'] - a_stats['off_eff']
        deff_diff = h_stats['def_eff'] - a_stats['def_eff']
        tov_diff = h_stats['tov_rate'] - a_stats['tov_rate']
        hc = 0 if neutral else 1

        # Skip if NaN
        if pd.isna([rating_diff, oeff_diff, deff_diff, tov_diff]).any():
            continue

        feature_vector = [
            rating_diff,
            oeff_diff,
            deff_diff,
            tov_diff,
            hc,
            ratings[home],
            ratings[away],
            h_stats['off_eff'],
            a_stats['off_eff'],
            h_stats['def_eff'],
            a_stats['def_eff']
        ]

        features.append(feature_vector)
        targets.append(margin)

    X = np.array(features)
    y = np.array(targets)

    # Remove any NaN rows (safety check)
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X = X[~nan_mask]
        y = y[~nan_mask]

    # Train model
    ml_model = LinearRegression()
    ml_model.fit(X, y)

    return ml_model


def load_data():
    """Fetch everything fresh from CBBD (season games + team stats) and retrain
    the model on it. This is the entire body of what used to run once at import
    time -- now it's callable, so refresh() can re-run it on every page load
    instead of only on process start."""

    #Have to use requests since no python library for CBBD yet
    games_url = "https://api.collegebasketballdata.com/games?season=2026"
    #Convert the API response into a json then a dataframe for easy use
    games_response = requests.get(games_url, headers=headers)
    games_data = games_response.json()
    games_df = pd.DataFrame(games_data)
    games_df = games_df[needed_cols].copy()

    #Only care about games where one of the teams was D1
    games_df = games_df[games_df["homeTeam"].isin(d1_teams) | games_df["awayTeam"].isin(d1_teams)].reset_index(drop=True)

    #Completed games = final status
    completed = games_df[(games_df["status"] == "final")].reset_index(drop=True)

    #Upcoming games = scheduled but not yet played
    upcoming = games_df[(games_df["status"] != "final")].reset_index(drop=True)

    #Create the margin column in completed
    completed = completed.copy()
    completed["margin"] = completed["homePoints"] - completed["awayPoints"]

    #Using requests pull all the statistical data from the data set
    stats_url = "https://api.collegebasketballdata.com/stats/team/season?season=2026"
    #Convert the API response into a json then a dataframe for easy use
    stats_response = requests.get(stats_url, headers=headers)
    stats_data = stats_response.json()
    stats_df = pd.DataFrame(stats_data)
    team_stats = pd.json_normalize(stats_df["teamStats"])
    team_stats.columns = ["teamStats_" + c.replace(".", "_") for c in team_stats.columns]

    opp_stats = pd.json_normalize(stats_df["opponentStats"])
    opp_stats.columns = ["opponentStats_" + c.replace(".", "_") for c in opp_stats.columns]

    #Combine the stats back into one dataframe
    stats_df = pd.concat(
        [stats_df.drop(["teamStats", "opponentStats"], axis=1),
         team_stats, opp_stats],
        axis=1
    )

    # Create efficiency stats using the dataset
    stats_df["off_eff"] = stats_df["teamStats_points_total"] / stats_df["teamStats_possessions"]
    stats_df["def_eff"] = stats_df["opponentStats_points_total"] / stats_df["opponentStats_possessions"]
    stats_df["tov_rate"] = stats_df["teamStats_fourFactors_turnoverRatio"]

    # Keep only the stats/columns that I plan on using
    stats_clean = stats_df[useful].copy()

    #define what margin is
    #sort the dataframe to have a line of home and away teams
    #Build the design matrix
    df = completed.dropna(subset=["homeTeam", "awayTeam", "homePoints", "awayPoints"]).copy()
    df = df.dropna(subset=["margin"]).reset_index(drop=True)

    teams = sorted(set(df["homeTeam"]).union(df["awayTeam"]))
    #Home teams get a +1 value and -1 is for away
    #This setup allows for the regression to assign each team a numeric rating
    X = pd.DataFrame(0, index=np.arange(len(df)), columns=teams)
    for i, row in df.iterrows():
        X.loc[i, row["homeTeam"]] = 1    # +1 for home team
        X.loc[i, row["awayTeam"]] = -1   # -1 for away team

    #Add home court column
    X["home_court"] = 1

    X = X.fillna(0)

    #Time to fit the margin
    y = df["margin"]

    #Create the linear regression based on the margins
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    #Ensure home field is counted for in the team ratings
    coefs = pd.Series(model.coef_, index=X.columns)
    home_court = coefs["home_court"]
    ratings = coefs.drop("home_court")
    #Make the average team=0
    ratings -= ratings.mean()

    # Sort ratings index for clean dropdown in Flask
    ratings = ratings.sort_index()

    # Create D1-only rankings
    D1_rankings = pd.DataFrame({'team': ratings.index, 'rating': ratings.values})
    # Filter for only D1 teams
    D1_rankings = D1_rankings[D1_rankings['team'].isin(d1_teams)].reset_index(drop=True)
    # Sort by rating descending
    D1_rankings = D1_rankings.sort_values(by='rating', ascending=False).reset_index(drop=True)

    prediction_model = train_prediction_model(completed, ratings, stats_clean)

    return {
        "completed": completed,
        "upcoming": upcoming,
        "stats_clean": stats_clean,
        "ratings": ratings,
        "home_court": home_court,
        "D1_rankings": D1_rankings,
        "prediction_model": prediction_model,
    }


def refresh(force=False):
    """Re-fetch everything from CBBD and retrain the model, replacing this
    module's data in place. Call this before serving any page (see app.py)
    so the site always reflects current ratings/results instead of whatever
    was live when the process last started -- no redeploy required.

    Throttled to once every MIN_REFRESH_INTERVAL_SECONDS so back-to-back page
    loads (or several visitors at once) don't turn into a burst of redundant
    CBBD calls; pass force=True to bypass that (e.g. an admin refresh button)."""
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


#Prediction function - EXACTLY LIKE FOOTBALL
def predict_game(home, away, neutral_site=False):
    rating_diff = ratings[home] - ratings[away]
    #Pull the home and away team stats and compare them
    h_stats = stats_clean.loc[stats_clean["team"] == home].iloc[0]
    a_stats = stats_clean.loc[stats_clean["team"] == away].iloc[0]

    # Compute stat differences between the two teams
    oeff_diff = h_stats["off_eff"] - a_stats["off_eff"]
    deff_diff = h_stats["def_eff"] - a_stats["def_eff"]
    turnover_diff = h_stats["tov_rate"] - a_stats["tov_rate"]

    #Whether or not home court advntage is applied
    home_advantage = 0 if neutral_site else 1
    #Different weights of each
    #w_rating=0.7
    #w_oeff=0.125
   # w_deff=0.125
    #w_tov=0.05
    #margin=(w_rating*rating_diff+(w_oeff*oeff_diff)+(w_deff*deff_diff)+(w_tov*turnover_diff)+home_advantage)
    features = np.array([[
        rating_diff,
        oeff_diff,
        deff_diff,
        turnover_diff,
        home_advantage,
        ratings[home],
        ratings[away],
        h_stats['off_eff'],
        a_stats['off_eff'],
        h_stats['def_eff'],
        a_stats['def_eff']
    ]])

    margin=prediction_model.predict(features)[0]
    #Calculate winprobability
    prob = 1 / (1 + np.exp(-margin / 5))
    return margin, prob

#Function to get betting lines for today's games
def get_betting_lines(season=2026):

    # Get today's date in EST
    est = pytz.timezone('America/New_York')
    today_est = datetime.now(est)

    # Start of day in EST, then convert to UTC for API
    start_of_day_est = today_est.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day_est = today_est.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Convert EST to UTC (API expects UTC with Z)
    start_of_day_utc = start_of_day_est.astimezone(pytz.UTC)
    end_of_day_utc = end_of_day_est.astimezone(pytz.UTC)

    # Format as ISO 8601 with Z
    start_date = start_of_day_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_date = end_of_day_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Basketball lines endpoint
    lines_url = f"https://api.collegebasketballdata.com/lines?season={season}&startDateRange={start_date}&endDateRange={end_date}"

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

            # Look for dk spreads
            dk_spread = None
            for line in lines:
                if line.get("provider") in ("DraftKings", "Draft Kings", "Bovada"):
                    dk_spread = line.get("spread")
                    break

            if dk_spread is not None:
                betting_lines[(home, away)] = dk_spread

        return betting_lines

    except Exception as e:
        print(f"Error fetching betting lines: {e}")
        return {}

# EXACTLY LIKE FOOTBALL
def calculate_edge_highlight(model_margin, betting_spread):
    # Convert betting spread to match our model's convention
    # If betting spread is +7, that means home is favored by 7
    # If betting spread is -7, that means away is favored by 7
    # We need to flip it to match our model's convention (positive = home favored)
    betting_margin = -betting_spread

    # Calculate the difference
    difference = abs(model_margin - betting_margin)

    if difference >= 10:
        return 'edge-big'
    elif difference >= 5:
        return 'edge-medium'
    else:
        return None

#Get upcoming games in order to print them
def get_upcoming_predictions(conference=None):
    """
    Get predictions for today's games by fetching directly from API.
    Uses date range parameter to get games from 12:00 AM - 11:59 PM EST today.
    """

    # Calculate date range for TODAY only (12:00 AM - 11:59 PM EST)
    utc = pytz.UTC
    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)

    # Start: midnight TODAY in EST -> UTC
    start_of_today_est = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_today_utc = start_of_today_est.astimezone(utc)

    # End: 11:59 PM TODAY in EST -> UTC
    end_of_today_est = now_est.replace(hour=23, minute=59, second=59, microsecond=999999)
    end_of_today_utc = end_of_today_est.astimezone(utc)

    # Format as ISO 8601 with milliseconds
    start_date_str = start_of_today_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_date_str = end_of_today_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Fetch today's games directly from API
    games_url = f"https://api.collegebasketballdata.com/games?season=2026&startDateRange={start_date_str}&endDateRange={end_date_str}"

    print(f"[get_upcoming_predictions] Fetching today's games from API")
    print(f"  Date range: {start_date_str} to {end_date_str}")

    try:
        response = requests.get(games_url, headers=headers)
        response.raise_for_status()
        todays_games_data = response.json()
        todays_games = pd.DataFrame(todays_games_data)

        print(f"  API returned: {len(todays_games)} games")

        if len(todays_games) == 0:
            print(f"  No games scheduled for today")
            return pd.DataFrame()

        # Filter for D1 teams
        todays_games = todays_games[
            todays_games["homeTeam"].isin(d1_teams) |
            todays_games["awayTeam"].isin(d1_teams)
        ]

        print(f"  After D1 filter: {len(todays_games)} games")

        # Filter for non-final games only (upcoming)
        games_to_predict = todays_games[todays_games["status"] != "final"].copy()

        print(f"  Upcoming (non-final): {len(games_to_predict)} games")

    except Exception as e:
        print(f"  Error fetching today's games from API: {e}")
        return pd.DataFrame()

    # Convert to EST for sorting
    games_to_predict["startDate"] = pd.to_datetime(
        games_to_predict["startDate"], utc=True, errors="coerce"
    )
    games_to_predict["startDate_EST"] = games_to_predict["startDate"].dt.tz_convert("America/New_York")

    # Sort by game time
    games_to_predict = games_to_predict.sort_values("startDate_EST")

    # Filter by conference if specified
    if conference is not None and conference != "All":
        games_to_predict = games_to_predict[
            (games_to_predict["homeConference"] == conference) |
            (games_to_predict["awayConference"] == conference)
        ]

    # Fetch betting lines for today
    betting_lines = get_betting_lines(season=2026)

    predictions = []
    for _, game in games_to_predict.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]
        is_neutral = game.get("neutralSite", False)

        # Skip games where data is missing
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_clean["team"].values or away not in stats_clean["team"].values:
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
                betting_margin = -betting_spread
                spread_diff = round(margin - betting_margin, 1)

            predictions.append({
                "home": home,
                "away": away,
                "predicted_winner": winner,
                "margin": round(abs(margin), 2),
                "prob": round(prob * 100, 1) if margin > 0 else round((1 - prob) * 100, 1),
                "betting_spread": betting_spread,
                "edge_class": edge_class,
                "spread_diff": spread_diff,
                "neutral_site": is_neutral,
                "date": game["startDate_EST"].strftime("%Y-%m-%d") if pd.notna(game.get("startDate_EST")) else None
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    print(f"[get_upcoming_predictions] Returning {len(predictions)} predictions")

    return pd.DataFrame(predictions)
