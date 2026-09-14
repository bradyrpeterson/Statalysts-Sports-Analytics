"""Repair the 2026 picks that the 2026-08-30 mass snapshot recorded wrong.

Two things went wrong that week. The snapshot walked every unplayed game left in the
season instead of just the upcoming slate, so every game past week 1 was stored before
its betting line existed -- and pandas hands back NaN rather than None for a missing
float, so those rows landed as betting_spread: NaN, recommended: False rather than
being skipped. Separately, the weeks 1-4 gate cleared edge_class outright, which kept
early picks out of the results panel entirely instead of merely out of the record.

This script fixes both for games already played:

  * Games recorded before their line existed are rebuilt -- the model is refit on only
    the games completed before that week (predictor.ratings_as_of), then graded against
    that week's closing DraftKings line. A pick is judged by the numbers that were live
    when it was made, not by a model that has since seen the result.
  * Every 2026 pick from weeks 1-4 is marked provisional, so it shows on the homepage
    but stays out of the headline profit/ROI record.

Picks imported from prior seasons are left alone.

    python scripts/backfill_early_season.py            # dry run
    python scripts/backfill_early_season.py --apply
"""
import math
import sys
from collections import Counter

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from football import predictor as fp

fp.refresh(force=True)

SEASON = 2026


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _game_index():
    """(home, away) -> (week, neutral_site) for every completed 2026 regular-season game."""
    completed = fp.completed
    regular = completed[completed["seasonType"] == "regular"]
    index = {}
    for _, game in regular.iterrows():
        index[(game["homeTeam"], game["awayTeam"])] = (
            int(float(game["week"])),
            bool(game.get("neutralSite", False)),
        )
    return index


def _grade(data, home_points, away_points):
    """Straight-up and against-the-spread grades, mirroring tracking.settle_pending_picks."""
    actual_winner = data["home"] if home_points > away_points else data["away"]
    straight_up_correct = actual_winner == data["predicted_winner"]

    ats_result = None
    if data.get("recommended") and not _is_missing(data.get("betting_spread")):
        cover_margin = (home_points - away_points) + data["betting_spread"]
        if cover_margin == 0:
            ats_result = "push"
        else:
            ats_result = (
                "win" if ((cover_margin > 0) == (data["recommended_side"] == "home")) else "loss"
            )
    return straight_up_correct, ats_result


def main():
    apply_changes = "--apply" in sys.argv

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate("firebase_credentials.json"))
    db = firestore.client()

    games = _game_index()
    lines_by_week = {}
    ratings_by_week = {}

    docs = list(
        db.collection("tracked_picks").where(filter=FieldFilter("season", "==", SEASON)).stream()
    )
    print(f"{len(docs)} tracked pick(s) in the {SEASON} season\n")

    updates = []
    rebuilt = 0
    flagged = 0
    skipped_no_line = 0

    for doc in docs:
        data = doc.to_dict()
        key = (data.get("home"), data.get("away"))
        if key not in games:
            continue  # not a completed regular-season game we can place in a week
        week, neutral = games[key]

        change = {}
        provisional = week <= fp.MODEL_FULLY_TRAINED_MIN_WEEKS
        if bool(data.get("provisional")) != provisional:
            change["provisional"] = provisional
            flagged += 1

        # The results panel groups a slate by week; picks snapshotted before that field
        # existed would otherwise fall back to a single date and show only part of the week.
        if data.get("week") != week:
            change["week"] = week
            change["season_type"] = "regular"

        # Rebuild picks that were stored before their line existed.
        if _is_missing(data.get("betting_spread")) and data.get("status") == "final":
            if week not in lines_by_week:
                lines_by_week[week] = fp.get_betting_lines(week)
                ratings_by_week[week] = fp.ratings_as_of(week)
            spread = lines_by_week[week].get(key)
            ratings = ratings_by_week[week]

            if spread is None or key[0] not in ratings.index or key[1] not in ratings.index:
                # Nothing to grade it against -- at least clear the NaN so it reads as
                # "no line" instead of silently failing every comparison.
                change.update({"betting_spread": None, "edge": None, "recommended": False})
                skipped_no_line += 1
            else:
                margin, prob = fp.predict_game_with(ratings, key[0], key[1], neutral)
                edge_class = fp.calculate_edge_highlight(margin, spread)
                spread_diff = margin - (-spread)
                recommended = bool(edge_class)
                change.update({
                    "predicted_winner": key[0] if margin > 0 else key[1],
                    "model_margin": round(abs(margin), 2),
                    "win_prob": round((prob if margin > 0 else 1 - prob) * 100, 1),
                    "betting_spread": float(spread),
                    "edge": abs(round(spread_diff, 1)),
                    "recommended": recommended,
                    "recommended_side": (
                        ("home" if margin > -spread else "away") if recommended else None
                    ),
                    "rebuilt_as_of_week": week,
                })
                rebuilt += 1

        if not change:
            continue

        # Anything already settled needs regrading against whatever changed above.
        if data.get("status") == "final" and data.get("actual_home_score") is not None:
            merged = {**data, **change}
            straight_up_correct, ats_result = _grade(
                merged, merged["actual_home_score"], merged["actual_away_score"]
            )
            change["straight_up_correct"] = straight_up_correct
            change["ats_result"] = ats_result

        updates.append((doc.reference, change, week))

    by_week = Counter(week for _, _, week in updates)
    print(f"{len(updates)} doc(s) to update  (rebuilt {rebuilt}, provisional flag {flagged}, "
          f"no line available {skipped_no_line})")
    for week in sorted(by_week):
        print(f"  week {week}: {by_week[week]}")

    if not updates:
        print("\nNothing to do.")
        return

    if not apply_changes:
        print("\nDry run -- re-run with --apply to write these.")
        return

    for start in range(0, len(updates), 450):
        batch = db.batch()
        for ref, change, _ in updates[start:start + 450]:
            batch.update(ref, change)
        batch.commit()
        print(f"  wrote {min(start + 450, len(updates))}/{len(updates)}")

    print(f"\nUpdated {len(updates)} pick(s).")


if __name__ == "__main__":
    main()
