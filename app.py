# app.py - COMPLETELY FIXED VERSION
# Unified Flask app for Football and Basketball predictors

from flask import Flask, render_template, request, url_for, session, redirect, jsonify
import sys
import json
import os
import pandas as pd
import numpy as np
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime

# Add both sport folders to Python path
sys.path.append('./football')
sys.path.append('./basketball')

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

def refresh_predictors():
    """Re-pull data from CFBD/CBBD and retrain both models. Called at the top of
    every page that shows predictions so the site reflects final scores and new
    games on every visit instead of only whenever the process last restarted.
    Each predictor's own refresh() throttles itself, so calling this from
    multiple routes in the same few seconds is cheap."""
    if FOOTBALL_AVAILABLE:
        try:
            football_predictor.refresh()
        except Exception as e:
            print(f"Error refreshing football predictor: {e}")
    if BASKETBALL_AVAILABLE:
        try:
            basketball_predictor.refresh()
        except Exception as e:
            print(f"Error refreshing basketball predictor: {e}")

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
            import traceback
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

@app.route("/")
def index():
    """Landing page - shows featured pick and top 5 rankings preview (PUBLIC)"""
    # Check if user is logged in
    user_logged_in = False
    if 'user_id' in session:
        try:
            user_doc = db.collection("users").document(session["user_id"]).get()
            if user_doc.exists:
                user_logged_in = user_doc.to_dict().get("status") == "active"
        except:
            pass
    user_email = session.get('email', None)

    try:
        refresh_predictors()

        # Get today's basketball games
        basketball_preds = basketball_predictor.get_upcoming_predictions()
        
        # Get this week's football games  
        football_preds = football_predictor.get_upcoming_predictions()
        
        # Find the best pick (highest edge) from either sport
        featured_pick = None
        
        # Check basketball for high-edge games
        if len(basketball_preds) > 0:
            basketball_preds['abs_spread_diff'] = basketball_preds['spread_diff'].abs()
            best_bball = basketball_preds.nlargest(1, 'abs_spread_diff')
            if len(best_bball) > 0:
                pick = best_bball.iloc[0]
                if pick.get('spread_diff') is not None and not pd.isna(pick['spread_diff']) and abs(float(pick['spread_diff'])) >= 5:
                    # Check if prob is already a percentage (>1) or decimal (0-1)
                    prob_value = float(pick['prob'])
                    if prob_value <= 1:
                        prob_value = prob_value * 100  # Convert from decimal to percentage
                    
                    # Handle betting_spread NaN
                    betting_spread_value = None if pd.isna(pick.get('betting_spread')) else pick.get('betting_spread')
                    
                    featured_pick = {
                        'sport': 'basketball',
                        'home': pick['home'],
                        'away': pick['away'],
                        'predicted_winner': pick['predicted_winner'],
                        'margin': round(float(pick['margin']), 1),
                        'prob': round(prob_value, 1),
                        'betting_spread': betting_spread_value,
                        'edge': round(abs(float(pick['spread_diff'])), 1),
                        'edge_class': pick.get('edge_class', ''),
                        'neutral_site': pick.get('neutral_site', False)
                    }
        
        # Check football if no basketball pick
        if not featured_pick and len(football_preds) > 0:
            football_preds['abs_spread_diff'] = football_preds['spread_diff'].abs()
            best_football = football_preds.nlargest(1, 'abs_spread_diff')
            if len(best_football) > 0:
                pick = best_football.iloc[0]
                if pick.get('spread_diff') is not None and not pd.isna(pick['spread_diff']) and abs(float(pick['spread_diff'])) >= 3:
                    # Check if prob is already a percentage (>1) or decimal (0-1)
                    prob_value = float(pick['prob'])
                    if prob_value <= 1:
                        prob_value = prob_value * 100  # Convert from decimal to percentage
                    
                    # Handle betting_spread NaN
                    betting_spread_value = None if pd.isna(pick.get('betting_spread')) else pick.get('betting_spread')
                    
                    featured_pick = {
                        'sport': 'football',
                        'home': pick['home'],
                        'away': pick['away'],
                        'predicted_winner': pick['predicted_winner'],
                        'margin': round(float(pick['margin']), 1),
                        'prob': round(prob_value, 1),
                        'betting_spread': betting_spread_value,
                        'edge': round(abs(float(pick['spread_diff'])), 1),
                        'edge_class': pick.get('edge_class', ''),
                        'neutral_site': pick.get('neutral_site', False)
                    }
        
        if not featured_pick:
            print("No featured pick found (no games with sufficient edge)")
        
        # Get top 5 rankings for preview
        football_top10 = football_predictor.FBS_rankings.head(10).to_dict('records')
        basketball_top10 = basketball_predictor.D1_rankings.head(10).to_dict('records')

        # Real track record for the homepage (falls back to empty record until picks are settled)
        try:
            overall_record = tracking.get_track_record(db)
            football_record = tracking.get_track_record(db, sport="football")
            basketball_record = tracking.get_track_record(db, sport="basketball")
        except Exception as e:
            print(f"Error loading track record: {e}")
            overall_record = {"total_picks": 0, "straight_up_win_pct": None, "recommended_count": 0,
                               "ats_win_pct": None, "profit_series": [], "total_profit": 0, "roi_pct": None}
            football_record = dict(overall_record)
            basketball_record = dict(overall_record)

        try:
            recent_results = tracking.get_recent_results(db, sport="football", limit=12)
        except Exception as e:
            print(f"Error loading recent results: {e}")
            recent_results = []

        return render_template('index.html',
                             recent_results=recent_results,
                             team_logos=football_logos,
                             featured_pick=featured_pick,
                             football_top10=football_top10,
                             basketball_top10=basketball_top10,
                             basketball_logos=basketball_logos,
                             has_games=len(basketball_preds) > 0 or len(football_preds) > 0,
                             user_logged_in=user_logged_in,
                             user_email=user_email,
                             overall_record=overall_record,
                             football_record=football_record,
                             basketball_record=basketball_record,
                             highlights=MODEL_HIGHLIGHTS,
                             football_model_fully_trained=football_predictor.model_fully_trained)
    except Exception as e:
        print(f"Error loading index: {e}")
        import traceback
        traceback.print_exc()
        empty_record = {"total_picks": 0, "straight_up_win_pct": None, "recommended_count": 0,
                         "ats_win_pct": None, "profit_series": [], "total_profit": 0, "roi_pct": None}
        return render_template('index.html',
                             recent_results=[],
                             team_logos={},
                             featured_pick=None,
                             football_top10=[],
                             basketball_top10=[],
                             basketball_logos={},
                             has_games=False,
                             user_logged_in=user_logged_in,
                             user_email=user_email,
                             overall_record=empty_record,
                             football_record=dict(empty_record),
                             basketball_record=dict(empty_record),
                             highlights=MODEL_HIGHLIGHTS,
                             football_model_fully_trained=getattr(football_predictor, 'model_fully_trained', True) if FOOTBALL_AVAILABLE else True)

@app.route("/football")
@login_required
def football():
    """Football predictions page (LOGIN REQUIRED)"""
    try:
        refresh_predictors()

        week = request.args.get("week", str(football_predictor.next_week))
        conference = request.args.get("conference", "All")
        
        # Convert week to int for predictor
        week_param = None
        if week != "All" and week != "bowl":
            try:
                week_param = int(week)
            except:
                week_param = football_predictor.next_week
        elif week == "bowl":
            week_param = "bowl"
        
        predictions_df = football_predictor.get_upcoming_predictions(
            week=week_param,
            conference=conference if conference != "All" else None
        )
        
        # COMPLETE FIX: Clean up ALL NaN values
        if len(predictions_df) > 0:
            # Check first row to see if prob is decimal or percentage
            sample_prob = predictions_df['prob'].iloc[0]
            if sample_prob <= 1:
                predictions_df['prob'] = predictions_df['prob'] * 100  # Convert to percentage
            
            predictions_df['margin'] = predictions_df['margin'].round(1)
            
            # Replace NaN betting_spread with None
            predictions_df['betting_spread'] = predictions_df['betting_spread'].apply(
                lambda x: None if pd.isna(x) else x
            )
            
            # Replace NaN spread_diff (edge) with None
            predictions_df['spread_diff'] = predictions_df['spread_diff'].apply(
                lambda x: None if pd.isna(x) else x
            )
        
        predictions = predictions_df.to_dict('records') if len(predictions_df) > 0 else []
        
        # Get list of conferences
        try:
            conferences = sorted(set(
                list(football_predictor.completed['homeConference'].unique()) +
                list(football_predictor.completed['awayConference'].unique())
            ))
        except:
            conferences = football_conferences if football_conferences else []
        
        return render_template('football.html',
                             predictions=predictions,
                             selected_week=week,
                             conferences=conferences,
                             selected_conference=conference,
                             team_logos=football_logos,
                             team_colors=football_colors,
                             model_fully_trained=football_predictor.model_fully_trained,
                             weeks_completed=football_predictor.weeks_completed)
    except Exception as e:
        print(f"Error loading football: {e}")
        import traceback
        traceback.print_exc()
        return render_template('football.html',
                             predictions=[],
                             conferences=football_conferences,
                             selected_week=str(football_predictor.next_week) if FOOTBALL_AVAILABLE else "1",
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
        refresh_predictors()

        conference = request.args.get("conference", "All")

        predictions_df = basketball_predictor.get_upcoming_predictions(
            conference=conference if conference != "All" else None
        )
        
        #Clean up ALL NaN values
        if len(predictions_df) > 0:
            # Check first row to see if prob is decimal or percentage
            sample_prob = predictions_df['prob'].iloc[0]
            if sample_prob <= 1:
                predictions_df['prob'] = predictions_df['prob'] * 100  # Convert to percentage
            
            predictions_df['margin'] = predictions_df['margin'].round(1)
            
            # Replace NaN betting_spread with None
            predictions_df['betting_spread'] = predictions_df['betting_spread'].apply(
                lambda x: None if pd.isna(x) else x
            )
            
            # Replace NaN spread_diff (edge) with None
            predictions_df['spread_diff'] = predictions_df['spread_diff'].apply(
                lambda x: None if pd.isna(x) else x
            )
        
        predictions = predictions_df.to_dict('records') if len(predictions_df) > 0 else []
        
        # Get list of conferences
        try:
            conferences = sorted(set(
                list(basketball_predictor.completed['homeConference'].unique()) +
                list(basketball_predictor.completed['awayConference'].unique())
            ))
        except:
            conferences = basketball_conferences if basketball_conferences else []
        
        return render_template('basketball.html',
                             predictions=predictions,
                             conferences=conferences,
                             selected_conference=conference)
    except Exception as e:
        print(f"Error loading basketball: {e}")
        import traceback
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
        refresh_predictors()

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
    user_logged_in = False
    if 'user_id' in session:
        try:
            user_doc = db.collection("users").document(session["user_id"]).get()
            if user_doc.exists:
                user_logged_in = user_doc.to_dict().get("status") == "active"
        except:
            pass
    user_email = session.get('email', None)

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
                         user_email=user_email,
                         predictions=sample_predictions)

@app.route("/matchup")
@login_required
def matchup():
    """Unified matchup predictor -- pick a sport, pick two teams, get a number.
    Replaces the old /football/custom and /basketball/custom pages, which now
    redirect here."""
    refresh_predictors()

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
    user_logged_in = False
    if 'user_id' in session:
        try:
            user_doc = db.collection("users").document(session["user_id"]).get()
            if user_doc.exists:
                user_logged_in = user_doc.to_dict().get("status") == "active"
        except:
            pass
    user_email = session.get('email', None)
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
    except Exception as e:
        print(f"[tracking] Daily job failed: {e}")
        import traceback
        traceback.print_exc()

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
        scheduler.add_job(run_daily_tracking_job, "cron", hour=6, minute=0)
        scheduler.start()

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


@app.route("/payment-pending")
def payment_pending():
    if 'user_id' not in session:
        return redirect('/login')
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=session.get("email"),
            client_reference_id=session.get("user_id"),
            success_url=request.host_url + "stripe-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "logout",  # logs them out on cancel
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        print(f"Stripe error: {e}")
        return redirect('/login')


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Create a Stripe Checkout session for $5/month subscription"""
    if 'user_id' not in session:
        return redirect('/login')

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": STRIPE_PRICE_ID,
                "quantity": 1,
            }],
            customer_email=session.get("email"),
            client_reference_id=session.get("user_id"),  # Firebase UID - used in webhook
            success_url=request.host_url + "stripe-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "logout",  # logs them out on cancel
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        print(f"Stripe error: {e}")
        return redirect("/payment-pending")


@app.route("/stripe-success")
def stripe_success():
    session_id = request.args.get('session_id')
    
    if session_id and 'user_id' in session:
        try:
            from datetime import timezone, timedelta
            
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

    from datetime import timezone, timedelta

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

    from datetime import timezone
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
    print("=" * 80)
    print("SPORTS ANALYTICS HUB - COMPLETELY FIXED VERSION")
    print("=" * 80)
    print(f"Football Predictor: {'✓ Available' if FOOTBALL_AVAILABLE else '✗ Not Available'}")
    print(f"Basketball Predictor: {'✓ Available' if BASKETBALL_AVAILABLE else '✗ Not Available'}")
    print("=" * 80)
    print("Starting server")
    print("=" * 80)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)