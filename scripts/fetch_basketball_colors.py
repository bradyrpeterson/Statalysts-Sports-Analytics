"""Build basketball/team_color.json from the CBBD /teams endpoint.

Mirrors football/team_color.json: a flat {school: "#rrggbb"} map keyed by the
same school names the ratings use, so the win-probability donut can draw each
team's arc in its own color.

Schools the API has no primaryColor for keep whatever the existing file has, so
a hand-filled entry survives a re-run. Anything still uncolored is left out
rather than guessed at -- the templates fall back to a neutral chart color.

Run from the repo root:  python scripts/fetch_basketball_colors.py
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

TEAMS_URL = "https://api.collegebasketballdata.com/teams"
D1_FILE = os.path.join(ROOT, "basketball", "d1_teams_2026.json")
OUT_FILE = os.path.join(ROOT, "basketball", "team_color.json")


def best_entry(entries):
    """A school name can appear more than once (Holy Cross is both a Patriot
    League school and an unrelated NAIA one). Prefer the entry that actually
    has a conference and a color."""
    return sorted(
        entries,
        key=lambda t: (t.get("conference") is None, not t.get("primaryColor")),
    )[0]


def main():
    api_key = os.getenv("API_KEY")
    if not api_key:
        sys.exit("API_KEY missing from .env")

    response = requests.get(TEAMS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    response.raise_for_status()

    by_school = {}
    for team in response.json():
        by_school.setdefault(team["school"], []).append(team)

    with open(D1_FILE) as f:
        d1_teams = json.load(f)

    existing = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            existing = json.load(f)

    colors, kept, uncolored = {}, [], []
    for school in d1_teams:
        entries = by_school.get(school)
        color = best_entry(entries).get("primaryColor") if entries else None
        if color:
            colors[school] = "#" + color.lstrip("#").lower()
        elif school in existing:
            colors[school] = existing[school]
            kept.append(school)
        else:
            uncolored.append(school)

    with open(OUT_FILE, "w") as f:
        json.dump(dict(sorted(colors.items())), f, indent=2)
        f.write("\n")

    print(f"Wrote {len(colors)} colors to {os.path.relpath(OUT_FILE, ROOT)}")
    if kept:
        print(f"Kept {len(kept)} existing entries the API has no color for: {', '.join(kept)}")
    if uncolored:
        print(f"No color available for {len(uncolored)}: {', '.join(uncolored)}")


if __name__ == "__main__":
    main()
