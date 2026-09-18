"""Add FCS teams to football/team_logos.json and football/team_color.json.

FBS-vs-FCS games are shown on the football page, so the FCS side needs a logo
and a color like everyone else. Pulls the CFBD /teams roster and writes each FCS
school in the same shape as the existing FBS entries -- the ESPN logo URL keyed
on CFBD's team id (the ids are shared) and a "#rrggbb" primary color.

Existing entries are never overwritten, so a hand-fixed FBS logo or color
survives a re-run. Run from the repo root:  python scripts/fetch_fcs_team_assets.py
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

TEAMS_URL = "https://api.collegefootballdata.com/teams"
LOGO_FILE = os.path.join(ROOT, "football", "team_logos.json")
COLOR_FILE = os.path.join(ROOT, "football", "team_color.json")
LOGO_URL = "http://a.espncdn.com/i/teamlogos/ncaa/500/{id}.png"
SEASON = 2026


def main():
    api_key = os.getenv("API_KEY")
    if not api_key:
        sys.exit("API_KEY not set")

    resp = requests.get(TEAMS_URL, params={"year": SEASON},
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    resp.raise_for_status()
    fcs = [t for t in resp.json() if (t.get("classification") or "").lower() == "fcs"]

    with open(LOGO_FILE) as f:
        logos = json.load(f)
    with open(COLOR_FILE) as f:
        colors = json.load(f)

    added_logos = added_colors = 0
    for t in fcs:
        school = t["school"]
        if school not in logos and t.get("id") and t.get("logos"):
            logos[school] = LOGO_URL.format(id=t["id"])
            added_logos += 1
        if school not in colors and t.get("color"):
            colors[school] = t["color"].lower()
            added_colors += 1

    for path, data in ((LOGO_FILE, logos), (COLOR_FILE, colors)):
        with open(path, "w") as f:
            json.dump(dict(sorted(data.items())), f, indent=2)
            f.write("\n")

    print(f"{len(fcs)} FCS teams: added {added_logos} logos, {added_colors} colors")


if __name__ == "__main__":
    main()
