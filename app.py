# Unified Flask app for the football and basketball predictors.

from flask import Flask, render_template, request, url_for, session, redirect, jsonify
import sys
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from functools import wraps

import pandas as pd
import pytz
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Add both sport folders to Python path
sys.path.append('./football')
sys.path.append('./basketball')

#Python block-buffers stdout when it isn't a terminal, so under gunicorn a worker's
#prints only reached the log when the buffer filled or the worker died -- hours late
#and out of order with gunicorn's own lines. Flush every line as it's written.
sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY',"asdfaDFdf23423@#@!$!@#@!$#@!$#@!$#@!")

# Initialize Firebase Admin
if not firebase_admin._apps:
    if os.path.exists('firebase_credentials.json'):
        cred = credentials.Certificate('firebase_credentials.json')
    else:
        service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
        cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

#How often the background thread re-pulls data and refits both models, and how soon
#it tries again after a failed pull (an API outage shouldn't leave stale data for an hour).
DATA_REFRESH_SECONDS = 3600
DATA_RETRY_SECONDS = 300
#How long a page waits for the very first load after a deploy or restart before
#rendering its empty fallback instead.
DATA_WAIT_SECONDS = 60

_predictors_loaded = threading.Event()


def refresh_predictors():
    """Re-pull data from CFBD/CBBD and retrain both models. Returns True if both
    loaded cleanly.

    Runs only on the background thread below, never inside a request. It used to be
    called at the top of every page, so once an hour -- and every time a worker was
    replaced -- some visitor's request did the whole download and refit, and on
    Render's small instances that ran past gunicorn's timeout and got the worker killed.
    """
    ok = True
    if FOOTBALL_AVAILABLE:
        try:
            football_predictor.refresh(force=True)
        except Exception as e:
            ok = False
            print(f"Error refreshing football predictor: {e}")
    if BASKETBALL_AVAILABLE:
        try:
            basketball_predictor.refresh(force=True)
        except Exception as e:
            ok = False
            print(f"Error refreshing basketball predictor: {e}")
    return ok


def _keep_predictors_fresh():
    while True:
        ok = refresh_predictors()
        #Set even after a failed first load so pages fall back to their empty state
        #rather than waiting on data that isn't coming.
        _predictors_loaded.set()
        time.sleep(DATA_REFRESH_SECONDS if ok else DATA_RETRY_SECONDS)


def wait_for_predictors():
    """Hold a request until the first data load has finished (only matters right
    after a restart). Later refreshes swap data in without blocking anyone."""
    return _predictors_loaded.wait(timeout=DATA_WAIT_SECONDS)

def current_user_state():
    """(logged_in, email) for the public pages, which render a signed-in header but
    do not require an account. Every public route needs the same two values."""
    user_logged_in = False
    if 'user_id' in session:
        try:
            user_doc = db.collection("users").document(session["user_id"]).get()
            if user_doc.exists:
                user_logged_in = user_doc.to_dict().get("status") == "active"
        except Exception as e:
            print(f"Error reading user state: {e}")
    return user_logged_in, session.get('email')


def clean_predictions(predictions_df):
    """Turn a predictions frame into template-ready rows.

    Probabilities arrive either as a fraction or already as a percentage depending on
    the predictor, and pandas stores a missing betting line as NaN, which Jinja renders
    as the string "nan" instead of falling through its `is not none` checks.
    """
    if len(predictions_df) == 0:
        return []

    predictions_df = predictions_df.copy()
    if predictions_df['prob'].iloc[0] <= 1:
        predictions_df['prob'] = predictions_df['prob'] * 100
    predictions_df['margin'] = predictions_df['margin'].round(1)
    for column in ('betting_spread', 'spread_diff'):
        predictions_df[column] = predictions_df[column].apply(
            lambda x: None if pd.isna(x) else x
        )
    return predictions_df.to_dict('records')


def conference_options(predictor, fallback):
    """Conferences seen in this season's completed games, or the static list if the
    predictor has not loaded any."""
    try:
        return sorted(set(
            list(predictor.completed['homeConference'].unique()) +
            list(predictor.completed['awayConference'].unique())
        ))
    except Exception:
        return fallback or []


def featured_pick_from(predictions_df, sport, min_edge):
    """The single biggest disagreement with the market, if it clears min_edge.

    The homepage shows one free pick; both sports pick theirs the same way and differ
    only in the threshold they have to clear.
    """
    if len(predictions_df) == 0:
        return None

    best = predictions_df.assign(
        abs_spread_diff=predictions_df['spread_diff'].abs()
    ).nlargest(1, 'abs_spread_diff')
    if len(best) == 0:
        return None

    pick = best.iloc[0]
    spread_diff = pick.get('spread_diff')
    if spread_diff is None or pd.isna(spread_diff) or abs(float(spread_diff)) < min_edge:
        return None

    prob = float(pick['prob'])
    if prob <= 1:
        prob *= 100
    return {
        'sport': sport,
        'home': pick['home'],
        'away': pick['away'],
        'predicted_winner': pick['predicted_winner'],
        'margin': round(float(pick['margin']), 1),
        'prob': round(prob, 1),
        'betting_spread': None if pd.isna(pick.get('betting_spread')) else pick.get('betting_spread'),
        'edge': round(abs(float(spread_diff)), 1),
        'edge_class': pick.get('edge_class', ''),
        'neutral_site': pick.get('neutral_site', False),
    }


def login_required(f):
    """Gate on having an account only. The paid-tier gate (subscription status/expiration
    checks) is dormant while access is free -- see /auth-callback, which now marks every
    new account "active" on creation. The Stripe plumbing (checkout, webhook, billing
    portal) is left in place, unused, for when a paid tier comes back."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if 'user_id' not in session:
                return redirect('/login')

            user_doc = db.collection("users").document(session["user_id"]).get()
            if not user_doc.exists:
                return redirect(url_for('login'))

            return f(*args, **kwargs)

        except Exception as e:
            print(f"Error in login_required: {e}")
            traceback.print_exc()
            return redirect(url_for('login'))
    return decorated_function


@app.route("/robots.txt")
def robots():
    """Crawlers request this on every visit; without it they generate 404 noise.
    Keeps bots off the authenticated and admin paths, which have nothing to index."""
    body = "\n".join([
        "User-agent: *",
        "Disallow: /admin/",
        "Disallow: /manage-subscription",
        "Disallow: /auth-callback",
        "Disallow: /webhook/",
        "Allow: /",
        "",
    ])
    return app.response_class(body, mimetype="text/plain")


@app.route("/login")
def login():
    """Login page"""
    return render_template("login.html")

@app.route("/auth-callback", methods=["POST"])
def auth_callback():
    data = request.get_json()
    uid = data.get("uid")
    email = data.get("email")

    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        # Access is free right now -- every new account gets full access immediately.
        user_ref.set({
            "email": email,
            "status": "active",
            "created_at": datetime.now(),
            "approved_at": datetime.now(),
            "subscription_expires": None
        })
        status = "active"
    else:
        status = user_doc.to_dict().get("status", "active")

    session["user_id"] = uid
    session["email"] = email

    return jsonify({'success': True, 'status': status}), 200

@app.route("/logout")
def logout():
    """Logout user"""
    session.clear()
    return render_template("logout.html")

# Try to import football predictor
try:
    from football import predictor as football_predictor
    
    football_dir = os.path.join(os.path.dirname(__file__), 'football')
    with open(os.path.join(football_dir, "fbs_teams_2026.json"), "r") as f:
        football_teams = json.load(f)
    with open(os.path.join(football_dir, "team_logos.json"), "r") as f:
        football_logos = json.load(f)
    with open(os.path.join(football_dir, "team_color.json"), "r") as f:
        football_colors = json.load(f)
    with open(os.path.join(football_dir, "conferences.json"), "r") as f:
        football_conferences = json.load(f)
    FOOTBALL_AVAILABLE = True
except Exception as e:
    print(f"Football predictor not available: {e}")
    FOOTBALL_AVAILABLE = False
    football_teams = []
    football_logos = {}
    football_colors = {}
    football_conferences = []

# Try to import basketball predictor
try:
    from basketball import predictor as basketball_predictor
    
    basketball_dir = os.path.join(os.path.dirname(__file__), 'basketball')
    with open(os.path.join(basketball_dir, "d1_teams_2026.json"), "r") as f:
        basketball_teams = json.load(f)
    with open(os.path.join(basketball_dir, "conferences.json"), "r") as f:
        basketball_conferences = json.load(f)
    with open(os.path.join(basketball_dir, "basketball_team_logos.json"), "r") as f:
        basketball_logos = json.load(f)
    with open(os.path.join(basketball_dir, "team_color.json"), "r") as f:
        basketball_colors = json.load(f)
    BASKETBALL_AVAILABLE = True
except Exception as e:
    print(f"Basketball predictor not available: {e}")
    BASKETBALL_AVAILABLE = False
    basketball_teams = []
    basketball_conferences = []
    basketball_logos = {}
    basketball_colors = {}

#Loading happens off the startup path: gunicorn kills a worker that takes longer than
#its timeout to boot, and downloading plus fitting both models can take that long on
#a small instance.
threading.Thread(target=_keep_predictors_fresh, daemon=True).start()

@app.route("/")
def index():
    """Landing page -- one free featured pick, a rankings preview and last week's
    graded results (PUBLIC)."""
    user_logged_in, _ = current_user_state()

    try:
        wait_for_predictors()

        basketball_preds = basketball_predictor.get_upcoming_predictions()
        #Current week only -- betting lines exist for nothing past it anyway.
        football_preds = football_predictor.get_upcoming_predictions(week=football_predictor.next_week)

        #Basketball gets first refusal on the featured slot, at a higher bar than
        #football: its lines move less, so a 3-point disagreement means less there.
        featured_pick = (
            featured_pick_from(basketball_preds, 'basketball', min_edge=5)
            or featured_pick_from(football_preds, 'football', min_edge=3)
        )
        if not featured_pick:
            print("No featured pick found (no games with sufficient edge)")

        try:
            recent_results = tracking.get_recent_results(db, sport="football")
        except Exception as e:
            print(f"Error loading recent results: {e}")
            recent_results = []

        return render_template('index.html',
                             recent_results=recent_results,
                             team_logos=football_logos,
                             featured_pick=featured_pick,
                             football_top10=football_predictor.FBS_rankings.head(10).to_dict('records'),
                             basketball_top10=basketball_predictor.D1_rankings.head(10).to_dict('records'),
                             basketball_logos=basketball_logos,
                             user_logged_in=user_logged_in,
                             highlights=MODEL_HIGHLIGHTS,
                             football_model_fully_trained=football_predictor.model_fully_trained)
    except Exception as e:
        print(f"Error loading index: {e}")
        traceback.print_exc()
        return render_template('index.html',
                             recent_results=[],
                             team_logos={},
                             featured_pick=None,
                             football_top10=[],
                             basketball_top10=[],
                             basketball_logos={},
                             user_logged_in=user_logged_in,
                             highlights=MODEL_HIGHLIGHTS,
                             football_model_fully_trained=getattr(football_predictor, 'model_fully_trained', True) if FOOTBALL_AVAILABLE else True)

@app.route("/football")
@login_required
def football():
    """Football predictions page (LOGIN REQUIRED)"""
    try:
        wait_for_predictors()

        week = request.args.get("week", str(football_predictor.next_week))
        conference = request.args.get("conference", "All")

        week_param = None
        if week == "bowl":
            week_param = "bowl"
        elif week != "All":
            try:
                week_param = int(week)
            except ValueError:
                week_param = football_predictor.next_week

        predictions_df = football_predictor.get_upcoming_predictions(
            week=week_param,
            conference=conference if conference != "All" else None
        )

        return render_template('football.html',
                             predictions=clean_predictions(predictions_df),
                             selected_week=week,
                             conferences=conference_options(football_predictor, football_conferences),
                             selected_conference=conference,
                             team_logos=football_logos,
                             team_colors=football_colors,
                             model_fully_trained=football_predictor.model_fully_trained,
                             weeks_completed=football_predictor.weeks_completed)
    except Exception as e:
        print(f"Error loading football: {e}")
        traceback.print_exc()
        return render_template('football.html',
                             predictions=[],
                             conferences=football_conferences,
                             selected_week=str(getattr(football_predictor, 'next_week', 1)) if FOOTBALL_AVAILABLE else "1",
                             selected_conference="All",
                             team_logos=football_logos,
                             team_colors=football_colors,
                             model_fully_trained=getattr(football_predictor, 'model_fully_trained', True) if FOOTBALL_AVAILABLE else True,
                             weeks_completed=getattr(football_predictor, 'weeks_completed', 0) if FOOTBALL_AVAILABLE else 0)

@app.route("/basketball")
@login_required
def basketball():
    """Basketball predictions page (LOGIN REQUIRED)"""
    try:
        wait_for_predictors()

        conference = request.args.get("conference", "All")
        predictions_df = basketball_predictor.get_upcoming_predictions(
            conference=conference if conference != "All" else None
        )

        return render_template('basketball.html',
                             predictions=clean_predictions(predictions_df),
                             conferences=conference_options(basketball_predictor, basketball_conferences),
                             selected_conference=conference)
    except Exception as e:
        print(f"Error loading basketball: {e}")
        traceback.print_exc()
        return render_template('basketball.html',
                             predictions=[],
                             conferences=basketball_conferences,
                             selected_conference="All")

@app.route("/rankings")
@login_required
def rankings():
    """Rankings page (LOGIN REQUIRED)"""
    try:
        wait_for_predictors()

        football_rankings = football_predictor.FBS_rankings.head(25).to_dict('records')
        basketball_rankings = basketball_predictor.D1_rankings.head(25).to_dict('records')
        
        return render_template('rankings.html',
                             football_rankings=football_rankings,
                             basketball_rankings=basketball_rankings)
    except Exception as e:
        print(f"Error loading rankings: {e}")
        return render_template('rankings.html', 
                             football_rankings=[], 
                             basketball_rankings=[])
    
@app.route("/example")
def example():
    """Public worked example of a football predictions page, using fictional games.

    Exists so visitors can see exactly what the real /football page looks like and
    how to read it, even in the off-season when there are no real games to show.
    """
    user_logged_in, user_email = current_user_state()

    sample_predictions = [
        {
            "home": "Ohio State", "away": "Michigan", "predicted_winner": "Ohio State",
            "margin": 9.5, "betting_spread": -3.5, "spread_diff": 6.0,
            "prob": 78.4, "neutral_site": False,
        },
        {
            "home": "Texas", "away": "Oklahoma", "predicted_winner": "Texas",
            "margin": 6.0, "betting_spread": -2.0, "spread_diff": 4.0,
            "prob": 68.2, "neutral_site": False,
        },
        {
            "home": "Georgia", "away": "Alabama", "predicted_winner": "Georgia",
            "margin": 3.0, "betting_spread": -2.5, "spread_diff": 0.5,
            "prob": 57.9, "neutral_site": False,
        },
        {
            "home": "Notre Dame", "away": "USC", "predicted_winner": "Notre Dame",
            "margin": 4.5, "betting_spread": None, "spread_diff": None,
            "prob": 64.1, "neutral_site": True,
        },
    ]

    return render_template('example.html',
                         user_logged_in=user_logged_in,
                         predictions=sample_predictions)

@app.route("/matchup")
@login_required
def matchup():
    """Unified matchup predictor -- pick a sport, pick two teams, get a number.
    Replaces the old /football/custom and /basketball/custom pages, which now
    redirect here."""
    wait_for_predictors()

    sports = {}
    if FOOTBALL_AVAILABLE:
        try:
            fbs = set(football_predictor.fbs_teams)
            sports["football"] = [
                {"name": t,
                 "logo": football_logos.get(t, ""),
                 "color": football_colors.get(t, "")}
                for t in sorted(football_predictor.ratings.index) if t in fbs
            ]
        except Exception as e:
            print(f"Error building football matchup teams: {e}")
    if BASKETBALL_AVAILABLE:
        try:
            d1 = set(basketball_predictor.d1_teams)
            sports["basketball"] = [
                {"name": t,
                 "logo": basketball_logos.get(t, ""),
                 "color": basketball_colors.get(t, "")}
                for t in sorted(basketball_predictor.ratings.index) if t in d1
            ]
        except Exception as e:
            print(f"Error building basketball matchup teams: {e}")

    requested = (request.args.get("sport") or "").lower()
    if requested in sports:
        default_sport = requested
    elif sports:
        default_sport = "football" if "football" in sports else next(iter(sports))
    else:
        default_sport = ""

    return render_template("matchup.html", sports=sports, default_sport=default_sport)

@app.route("/football/custom")
def football_custom():
    """Old football-only predictor; folded into the unified /matchup page."""
    return redirect("/matchup?sport=football")

@app.route("/basketball/custom")
def basketball_custom():
    """Old basketball-only predictor; folded into the unified /matchup page."""
    return redirect("/matchup?sport=basketball")

@app.route("/api/predict", methods=["POST"])
@login_required
def predict_matchup():
    """Run one hypothetical game through whichever sport's model was asked for."""
    data = request.json or {}
    sport = (data.get("sport") or "").lower()
    home = data.get("home")
    away = data.get("away")
    neutral = bool(data.get("neutral", False))

    if sport == "football" and FOOTBALL_AVAILABLE:
        model = football_predictor
    elif sport == "basketball" and BASKETBALL_AVAILABLE:
        model = basketball_predictor
    else:
        return jsonify({"success": False, "error": "That sport isn't available right now."}), 400

    if not home or not away:
        return jsonify({"success": False, "error": "Pick both teams."}), 400
    if home == away:
        return jsonify({"success": False, "error": "Pick two different teams."}), 400

    try:
        margin, prob = model.predict_game(home, away, neutral_site=neutral)
    except KeyError as e:
        return jsonify({"success": False,
                        "error": f"No rating for {str(e).strip(chr(39))} yet."}), 400
    except IndexError:
        return jsonify({"success": False,
                        "error": "One of those teams has no game data yet this season."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    margin = float(margin)
    prob = float(prob)
    # predict_game returns the probability for the home team, sometimes as a
    # fraction and sometimes already as a percentage.
    if prob <= 1:
        prob *= 100

    return jsonify({
        "success": True,
        "winner": home if margin > 0 else away,
        "margin": round(abs(margin), 1),
        "home_prob": round(prob, 1),
        "away_prob": round(100 - prob, 1),
    })

@app.route("/terms")
def terms():
    user_logged_in, user_email = current_user_state()
    return render_template("terms.html",
                           user_logged_in=user_logged_in,
                           user_email=user_email)
import stripe

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID')
ADMIN_SECRET = os.environ.get('ADMIN_SECRET')

# Manually-curated bragging-rights callouts for the homepage track-record section.
# Fill these in yourself each season -- e.g. {"sport": "football", "text": "Ranked Ohio State #1 entering Week 1."}
MODEL_HIGHLIGHTS = []

# Track record: snapshot picks daily, settle them once games finish
import tracking

# The job settles the week's games; running it at 6am ET means Sunday's run lands after
# even the latest Saturday night kickoff has gone final.
TRACKING_JOB_HOUR = 6
TRACKING_JOB_STATE_DOC = "job_state/daily_tracking"


def _tracking_today():
    return datetime.now(pytz.timezone("America/New_York"))


def run_daily_tracking_job():
    try:
        snapshotted = tracking.snapshot_todays_picks(
            db,
            football_predictor=football_predictor if FOOTBALL_AVAILABLE else None,
            basketball_predictor=basketball_predictor if BASKETBALL_AVAILABLE else None,
        )
        settled = tracking.settle_pending_picks(
            db,
            football_predictor=football_predictor if FOOTBALL_AVAILABLE else None,
            basketball_predictor=basketball_predictor if BASKETBALL_AVAILABLE else None,
        )
        print(f"[tracking] Snapshotted {snapshotted} new picks, settled {settled} picks")
        #Recorded so a worker that starts up after a missed run can tell and catch up.
        db.document(TRACKING_JOB_STATE_DOC).set({
            "last_run_date": _tracking_today().date().isoformat(),
            "last_run_at": datetime.now(timezone.utc),
            "snapshotted": snapshotted,
            "settled": settled,
        })
    except Exception as e:
        print(f"[tracking] Daily job failed: {e}")
        traceback.print_exc()


def _run_tracking_job_if_missed():
    """Run the job now if today's scheduled time has passed without it running.

    The scheduler lives inside a gunicorn worker, and --max-requests recycles those
    workers regularly. A worker replaced across the 6am mark takes the job's next run
    to be tomorrow, so the day is skipped silently -- which on a Sunday means last
    week's results sit unsettled until the following week. Checking the last recorded
    run on startup closes that window.
    """
    try:
        #Snapshotting reads the models, so a catch-up that ran before they loaded would
        #save nothing and still mark the day as done.
        _predictors_loaded.wait()
        now = _tracking_today()
        if now.hour < TRACKING_JOB_HOUR:
            return  # today's run hasn't come due yet
        today = now.date().isoformat()
        state = db.document(TRACKING_JOB_STATE_DOC).get()
        if state.exists and (state.to_dict() or {}).get("last_run_date") == today:
            return
        print(f"[tracking] No run recorded for {today} -- catching up now")
        run_daily_tracking_job()
    except Exception as e:
        print(f"[tracking] Catch-up check failed: {e}")

def _claim_scheduler_slot():
    """Return a held lock file if this process should own the scheduler, else None.

    Gunicorn runs more than one worker, and each imports this module, so without a
    guard every worker would schedule its own copy of the tracking job. An flock is
    exclusive across processes on the instance and is released automatically if the
    holder dies, so a recycled worker can pick the job back up.
    """
    import fcntl

    lock_file = open("/tmp/statalysts-scheduler.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None
    return lock_file


if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    # Guard avoids double-scheduling under Flask's debug-mode reloader (which forks a child
    # process with WERKZEUG_RUN_MAIN=true); gunicorn in production never sets this var, so the
    # scheduler still starts there.
    # Module-level reference keeps the lock's file descriptor open for the process lifetime.
    _scheduler_lock = _claim_scheduler_slot()
    if _scheduler_lock is not None:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(timezone="America/New_York")
        #misfire_grace_time keeps a run that fires late (a busy worker, a slow CFBD
        #call on the previous job) from being dropped outright.
        scheduler.add_job(
            run_daily_tracking_job, "cron",
            hour=TRACKING_JOB_HOUR, minute=0, misfire_grace_time=3600,
        )
        scheduler.start()
        #Off the request path: a missed day is caught up without delaying startup.
        threading.Thread(target=_run_tracking_job_if_missed, daemon=True).start()

@app.route("/admin/settle-picks", methods=["POST"])
def admin_settle_picks():
    """Manually trigger the snapshot+settle job (testing/recovery). POST {"secret": "..."}"""
    data = request.get_json(silent=True) or {}
    if data.get("secret") != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    snapshotted = tracking.snapshot_todays_picks(
        db,
        football_predictor=football_predictor if FOOTBALL_AVAILABLE else None,
        basketball_predictor=basketball_predictor if BASKETBALL_AVAILABLE else None,
    )
    settled = tracking.settle_pending_picks(
        db,
        football_predictor=football_predictor if FOOTBALL_AVAILABLE else None,
        basketball_predictor=basketball_predictor if BASKETBALL_AVAILABLE else None,
    )
    return jsonify({"success": True, "snapshotted": snapshotted, "settled": settled}), 200


def _start_checkout(on_error):
    """Open a Stripe Checkout session for the signed-in user.

    Both checkout entry points build an identical session and differ only in where
    they send someone when Stripe refuses.
    """
    if 'user_id' not in session:
        return redirect('/login')
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=session.get("email"),
            client_reference_id=session.get("user_id"),  # Firebase UID -- read by the webhook
            success_url=request.host_url + "stripe-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "logout",  # logs them out on cancel
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        print(f"Stripe error: {e}")
        return redirect(on_error)


@app.route("/payment-pending")
def payment_pending():
    return _start_checkout(on_error='/login')


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Start the $5/month subscription checkout."""
    return _start_checkout(on_error='/payment-pending')


@app.route("/stripe-success")
def stripe_success():
    session_id = request.args.get('session_id')
    
    if session_id and 'user_id' in session:
        try:
            # Verify payment directly with Stripe — don't wait for webhook
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            
            if checkout_session.payment_status == 'paid':
                user_ref = db.collection("users").document(session['user_id'])
                user_ref.update({
                    "status": "active",
                    "subscription_id": checkout_session.subscription,
                    "approved_at": datetime.now(timezone.utc),
                    "subscription_expires": datetime.now(timezone.utc) + timedelta(days=35),
                })
                print(f"Activated via success redirect for UID: {session['user_id']}")
        except Exception as e:
            print(f"Error verifying Stripe session: {e}")
    
    return redirect("/")


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        print(f"Webhook error: {e}")
        return jsonify({"error": "Invalid signature"}), 400

    if event["type"] == "checkout.session.completed":
        session_data = event["data"]["object"]
        uid = session_data["client_reference_id"]
        subscription_id = session_data["subscription"]
        customer_id = session_data["customer"]

        if uid:
            user_ref = db.collection("users").document(uid)
            user_ref.update({
                "status": "active",
                "subscription_id": subscription_id,
                "stripe_customer_id": customer_id,
                "approved_at": datetime.now(timezone.utc),
                "subscription_expires": datetime.now(timezone.utc) + timedelta(days=35),
            })
            print(f"Activated account for UID: {uid}")

    elif event["type"] == "invoice.paid":
        invoice = event["data"]["object"]
        customer_email = invoice["customer_email"]

        if customer_email:
            users = db.collection("users").where(filter=FieldFilter("email", "==", customer_email)).get()
            for user_doc in users:
                user_doc.reference.update({
                    "status": "active",
                    "subscription_expires": datetime.now(timezone.utc) + timedelta(days=35),
                })
                print(f"Renewed subscription for: {customer_email}")

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        sub_id = subscription["id"]

        users = db.collection("users").where(filter=FieldFilter("subscription_id", "==", sub_id)).get()
        for user_doc in users:
            user_doc.reference.update({"status": "expired"})
            print(f"Cancelled subscription: {sub_id}")

    return jsonify({"status": "ok"}), 200

@app.route("/manage-subscription")
@login_required
def manage_subscription():
    user_ref = db.collection("users").document(session["user_id"])
    user_data = user_ref.get().to_dict()

    customer_id = user_data.get("stripe_customer_id")

    if not customer_id:
        return redirect("/football")  # free account, nothing to manage

    

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.host_url + "football",
        )
        return redirect(portal_session.url, code=303)
    except Exception as e:
        print(f"Portal error: {e}")
        return redirect("/football")

@app.route("/admin/grant-access", methods=["POST"])
def admin_grant_access():
    """
    Grant free/admin access to an account.
    POST with JSON: {"email": "...", "secret": "...", "free": true}
    """
    data = request.get_json()

    if not data or data.get("secret") != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    email = data.get("email")
    is_free = data.get("free", True)  # True = never expires

    if not email:
        return jsonify({"error": "Email required"}), 400

    # Find user by email
    users = db.collection("users").where(filter=FieldFilter("email", "==", email)).get()

    if not users:
        return jsonify({"error": f"No user found with email {email}"}), 404

    for user_doc in users:
        update_data = {
            "status": "active",
            "approved_at": datetime.now(timezone.utc),
            "is_free": is_free,
        }
        if is_free:
            update_data["subscription_expires"] = None  # Never expires
        user_doc.reference.update(update_data)

    return jsonify({"success": True, "message": f"Access granted to {email}"}), 200

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    print(f"Football predictor: {'available' if FOOTBALL_AVAILABLE else 'NOT available'}")
    print(f"Basketball predictor: {'available' if BASKETBALL_AVAILABLE else 'NOT available'}")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)