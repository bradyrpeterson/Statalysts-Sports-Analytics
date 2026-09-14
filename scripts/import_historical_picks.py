"""One-off import of the 2025 football Week 9-10 manual tracker into tracked_picks.

Only Weeks 9 and 10 are imported: those are the only weeks in the source PDF
("CFP Predictor Tracker.pdf") with enough per-game data (model win%, projected
margin, closing spread, actual result) to grade honestly. Weeks 11-12 in that
PDF only have closing spread + actual result with no per-game model
projection, so they're skipped here rather than guessed at.

The tracker's "Closing Spread" column turned out to use the OPPOSITE sign
convention from this app's own betting_spread (verified empirically below:
negating it and grading "recommended_side = the model's straight-up pick"
reproduces the tracker's own stated Week 9 (12/20 ATS) and Week 10 (13/20 ATS)
totals exactly -- see verify_against_known_totals()).

Run once: python3 scripts/import_historical_picks.py
"""
import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import credentials, firestore
from football import predictor as football_predictor

SEASON = 2025

# (home, away, predicted_side, win_prob, model_margin, closing_spread, actual_side, actual_margin)
# predicted_side / actual_side are "home" or "away". closing_spread is exactly as it
# appears in the source tracker (home team's number, tracker's own sign convention).
WEEK_9 = [
    ("Oklahoma", "Ole Miss", "home", 60.12, 2.87, -4.5, "away", 8),
    ("Arkansas", "Auburn", "home", 56.0, 1.69, -2.5, "away", 9),
    ("South Carolina", "Alabama", "away", 87.57, 13.67, -12.5, "away", 7),
    ("Vanderbilt", "Missouri", "home", 74.96, 7.68, -3.5, "home", 7),
    ("Mississippi State", "Texas", "away", 58.63, 2.44, -8.5, "away", 7),
    ("LSU", "Texas A&M", "home", 51.08, 0.3, 2.5, "away", 24),
    ("Kentucky", "Tennessee", "away", 75.60, 7.92, -8.5, "away", 22),
    ("Indiana", "UCLA", "home", 96.26, 22.73, -24.5, "home", 50),
    ("Nebraska", "Northwestern", "home", 79.71, 9.58, -7.5, "home", 7),
    ("Purdue", "Rutgers", "away", 52.82, 0.79, 2.5, "away", 3),
    ("Washington", "Illinois", "home", 64.97, 4.32, -3.5, "home", 17),
    ("Iowa", "Minnesota", "home", 79.45, 7.89, -7.5, "home", 38),
    ("Oregon", "Wisconsin", "home", 98.66, 30.11, -31.5, "home", 14),
    ("Michigan State", "Michigan", "away", 81.45, 13.88, -13.5, "away", 11),
    ("North Carolina", "Virginia", "away", 95.33, 20.66, -12.5, "away", 1),
    ("Georgia Tech", "Syracuse", "home", 88.05, 13.98, -17.5, "home", 25),
    ("Wake Forest", "SMU", "home", 54.85, 1.36, 3.5, "home", 1),
    ("Pittsburgh", "NC State", "home", 75.22, 7.77, -6.5, "home", 19),
    ("Miami", "Stanford", "home", 97.15, 24.71, -29.5, "home", 35),
    ("Louisville", "Boston College", "home", 96.70, 23.65, -24.5, "home", 14),
]

WEEK_10 = [
    ("Texas", "Vanderbilt", "home", 55.01, 1.41, 3.5, "home", 3),
    ("Florida", "Georgia", "away", 63.72, 3.94, -6.5, "away", 4),
    ("Arkansas", "Mississippi State", "home", 57.45, 2.1, -5.5, "away", 3),
    ("Ole Miss", "South Carolina", "home", 83.81, 11.51, -13.5, "home", 16),
    ("Tennessee", "Oklahoma", "home", 68.82, 5.54, -2.5, "away", 6),
    ("Auburn", "Kentucky", "home", 82.74, 10.97, -11.5, "away", 7),
    ("Ohio State", "Penn State", "home", 92.71, 17.8, -17.5, "home", 24),
    ("Illinois", "Rutgers", "home", 81.8, 10.52, -12.5, "home", 22),
    ("Maryland", "Indiana", "away", 93.46, 19.3, -21.5, "away", 45),
    ("Minnesota", "Michigan State", "home", 52.16, 0.6, -5.5, "home", 3),
    ("Michigan", "Purdue", "home", 94.29, 19.63, -22.5, "home", 5),
    ("Nebraska", "USC", "away", 69.82, 5.87, -4.5, "away", 4),
    ("California", "Virginia", "away", 85.74, 12.56, -5.5, "away", 10),
    ("SMU", "Miami", "away", 81.99, 10.61, -6.5, "home", 6),
    ("Clemson", "Duke", "away", 51.7, 0.48, 4.5, "away", 1),
    ("Virginia Tech", "Louisville", "away", 79.74, 9.59, -10.5, "away", 12),
    ("Boston College", "Notre Dame", "away", 98.43, 28.96, -30.5, "away", 15),
    ("Stanford", "Pittsburgh", "away", 84.22, 14.72, -13.5, "away", 15),
    ("NC State", "Georgia Tech", "away", 61.10, 3.16, -4.5, "home", 12),
    ("Florida State", "Wake Forest", "home", 90.38, 15.68, -12.5, "home", 35),
]


def verify_against_known_totals():
    """Sanity check the grading formula against the tracker's own stated aggregates
    before writing anything, so a bad sign-convention assumption fails loudly here
    instead of silently corrupting the database."""
    for label, games, expected_wins in (("Week 9", WEEK_9, 12), ("Week 10", WEEK_10, 13)):
        wins = 0
        for home, away, pred_side, _, _, closing_spread, actual_side, actual_margin in games:
            actual_home_margin = actual_margin if actual_side == "home" else -actual_margin
            diff = actual_home_margin - closing_spread
            if diff == 0:
                continue
            home_covered = diff > 0
            picked_covered = (pred_side == "home") == home_covered
            if picked_covered:
                wins += 1
        assert wins == expected_wins, f"{label}: computed {wins} ATS wins, expected {expected_wins}"
        print(f"{label}: verified {wins}/{len(games)} ATS wins matches tracker's stated total")


def find_actual_game(week, home, away, actual_margin):
    """Cross-reference the real CFBD data already loaded in football_predictor to
    recover the true home/away orientation and exact date for this matchup."""
    candidates = football_predictor.completed[
        (football_predictor.completed["week"] == week)
        & (football_predictor.completed["season"] == SEASON)
    ]
    for _, game in candidates.iterrows():
        teams = {game["homeTeam"], game["awayTeam"]}
        if teams != {home, away}:
            continue
        margin = abs(game["homePoints"] - game["awayPoints"])
        if abs(margin - actual_margin) < 0.5:
            return game
    return None


def build_doc(week, home, away, pred_side, win_prob, model_margin, closing_spread, actual_side, actual_margin):
    real_game = find_actual_game(week, home, away, actual_margin)
    if real_game is None:
        print(f"  WARNING: could not confirm {home} vs {away} (week {week}) against real game data -- skipping")
        return None

    swapped = real_game["homeTeam"] != home  # our "home" guess from the "@ " parsing didn't match the API

    if swapped:
        home, away = away, home
        pred_side = "away" if pred_side == "home" else "home"
        actual_side = "away" if actual_side == "home" else "home"
        closing_spread = -closing_spread

    date_str = real_game["startDate"].strftime("%Y-%m-%d") if hasattr(real_game["startDate"], "strftime") else str(real_game["startDate"])[:10]

    model_home_margin = model_margin if pred_side == "home" else -model_margin
    betting_spread = -closing_spread  # convert tracker's convention to this app's (see module docstring)
    betting_home_margin = -betting_spread
    edge = round(abs(model_home_margin - betting_home_margin), 2)

    actual_home_margin = actual_margin if actual_side == "home" else -actual_margin
    cover_margin = actual_home_margin + betting_spread
    if cover_margin == 0:
        ats_result = "push"
    else:
        home_covered = cover_margin > 0
        ats_result = "win" if (pred_side == "home") == home_covered else "loss"

    straight_up_correct = pred_side == actual_side

    slug = lambda s: s.replace(" ", "-").replace("/", "-")
    doc_id = f"football_{date_str}_{slug(home)}_{slug(away)}"

    return doc_id, {
        "sport": "football",
        "date": date_str,
        "season": SEASON,
        "home": home,
        "away": away,
        "predicted_winner": home if pred_side == "home" else away,
        "model_margin": model_margin,
        "win_prob": win_prob,
        "betting_spread": betting_spread,
        "edge": edge,
        "recommended": True,
        "recommended_side": pred_side,
        "status": "final",
        "actual_home_score": None,
        "actual_away_score": None,
        "straight_up_correct": straight_up_correct,
        "ats_result": ats_result,
        "source": "historical_import",
        "created_at": datetime.now(timezone.utc),
        "settled_at": datetime.now(timezone.utc),
    }


def main():
    verify_against_known_totals()

    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    written = 0
    for week, games in ((9, WEEK_9), (10, WEEK_10)):
        for row in games:
            result = build_doc(week, *row)
            if result is None:
                continue
            doc_id, data = result
            db.collection("tracked_picks").document(doc_id).set(data)
            written += 1
    print(f"Wrote {written} historical picks")

    import tracking
    record = tracking.get_track_record(db, sport="football", season=SEASON)
    print("Post-import track record (football, 2025):", record)


if __name__ == "__main__":
    main()
