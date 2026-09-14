"""Clear tracked_picks rows that were snapshotted before their betting line existed.

On 2026-08-30 the daily job snapshotted every unplayed game left in the season in one
burst. Lines are only fetched for the current week, so every game past it was stored
with no spread -- and because pandas hands back NaN rather than None for a missing
value in a float column, those rows landed in Firestore as betting_spread: NaN,
edge: NaN, recommended: False instead of being skipped.

The snapshot is idempotent by doc id, so those rows would never be revisited: the whole
rest of the season was pre-committed as ungraded. tracking.py now snapshots one slate at
a time (SNAPSHOT_HORIZON_DAYS), but the stale rows have to be cleared before the weekly
snapshot can write good ones in their place.

Only *pending* rows are touched -- nothing with a score attached is removed. Games
already settled keep their result and their straight-up grade; they're excluded from the
against-the-spread record either way, which is what the weeks 1-4 policy calls for.

    python scripts/repair_tracked_picks.py            # dry run, prints what it would do
    python scripts/repair_tracked_picks.py --apply    # actually make the changes
"""
import math
import sys
from collections import Counter

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def main():
    apply_changes = "--apply" in sys.argv

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate("firebase_credentials.json"))
    db = firestore.client()

    pending = list(
        db.collection("tracked_picks")
        .where(filter=FieldFilter("status", "==", "pending"))
        .stream()
    )

    stale = [d for d in pending if _is_missing(d.to_dict().get("betting_spread"))]
    intact = len(pending) - len(stale)

    by_date = Counter(d.to_dict().get("date") for d in stale)
    print(f"{len(pending)} pending pick(s); {len(stale)} have no usable betting line")
    for date in sorted(by_date):
        print(f"  {date}  {by_date[date]}")
    print(f"leaving {intact} pending pick(s) that do have a line")

    if not stale:
        print("\nNothing to repair.")
        return

    if not apply_changes:
        print("\nDry run -- re-run with --apply to clear these.")
        return

    # Firestore caps a batch at 500 writes.
    for start in range(0, len(stale), 450):
        batch = db.batch()
        for doc in stale[start:start + 450]:
            batch.delete(doc.reference)
        batch.commit()
        print(f"  cleared {min(start + 450, len(stale))}/{len(stale)}")

    import tracking
    print(f"\nCleared {len(stale)} pick(s). The daily job will re-snapshot each slate "
          f"within {tracking.SNAPSHOT_HORIZON_DAYS} days of kickoff, with that week's line.")


if __name__ == "__main__":
    main()
