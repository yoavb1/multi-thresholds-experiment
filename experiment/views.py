from django.contrib.sessions.models import Session
from django.shortcuts import render, redirect
from django.http import HttpResponseForbidden, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import csv
import pandas as pd
import random
import datetime
import os
import uuid
import logging
from .models import *

# FileLock for atomic CSV operations - prevents race conditions
# Install: pip install filelock
from filelock import FileLock, Timeout

# Set up logging
logger = logging.getLogger(__name__)


def log_landing_attempt(request, aid, source):
    """
    Log EVERY landing page attempt to a CSV file for debugging.
    This runs BEFORE any database operations, so we can see all attempts.
    """
    try:
        log_path = os.path.join(settings.BASE_DIR, 'data', 'landing_attempts.csv')
        file_exists = os.path.exists(log_path)

        # Get all URL parameters for debugging
        all_params = dict(request.GET)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'aid', 'source', 'ip', 'user_agent', 'referer', 'all_params'])
            writer.writerow([
                datetime.datetime.now().isoformat(),
                aid,
                source,
                request.META.get('REMOTE_ADDR', 'unknown'),
                request.META.get('HTTP_USER_AGENT', 'unknown')[:100],
                request.META.get('HTTP_REFERER', 'none')[:100] if request.META.get('HTTP_REFERER') else 'none',
                str(all_params)[:200]
            ])
    except Exception as e:
        logger.error(f"Failed to log landing attempt: {e}")


def load_block_trials(csv_row_id=None) -> tuple:
    """
    Load trial data from CSV for a user.
    UPDATED: Now targeted at DATA/data.csv using vertical condition indexing.
    """
    STIMULI_SCALAR = 6.5

    # 1. Update the directory path and target name
    csv_path = os.path.join(settings.BASE_DIR, "data", "data.csv")
    lock_path = csv_path + ".lock"

    if not os.path.exists(csv_path):
        logger.error(f"CRITICAL: CSV file not found at {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Helper mapping to support 'signal', 'noise', and 'not sure'
    def get_ds_state(val):
        # Assumes values: 0/noise -> 'noise', 2/signal -> 'signal', 1/uncertain -> 'not sure'
        # Adjust integers below if your data.csv uses different keys (e.g., strings)
        try:
            v = int(val)
            mapping = {0: 'noise', 1: 'not sure', 2: 'signal'}
            return mapping.get(v, 'not sure')
        except (ValueError, TypeError):
            # Fallback if your CSV already lists text strings
            val_str = str(val).strip().lower()
            if 'signal' in val_str: return 'signal'
            if 'noise' in val_str: return 'noise'
            return 'not sure'

    # 2. Select condition row atomically using FileLock
    try:
        with FileLock(lock_path, timeout=30):
            event_data = pd.read_csv(csv_path)

            # Initialize a 'used' state column if it doesn't exist in your new data.csv
            if 'used' not in event_data.columns:
                event_data['used'] = 0

            if csv_row_id:
                selected_rows = event_data[event_data['condition_id'] == csv_row_id]
                row_id = csv_row_id
            else:
                # Find unassigned condition groups
                fresh_ids = event_data[event_data['used'] == 0]['condition_id'].unique()
                in_prog_ids = event_data[event_data['used'] == 0.5]['condition_id'].unique()

                if len(fresh_ids) > 0:
                    row_id = int(random.choice(fresh_ids))
                    event_data.loc[event_data['condition_id'] == row_id, 'used'] = 0.5
                    event_data.to_csv(csv_path, index=False)
                elif len(in_prog_ids) > 0:
                    row_id = int(random.choice(in_prog_ids))
                else:
                    row_id = int(random.choice(event_data['condition_id'].unique()))

            selected_rows = event_data[event_data['condition_id'] == row_id].sort_values('item_id')

    except Timeout:
        event_data = pd.read_csv(csv_path)
        row_id = int(random.choice(event_data['condition_id'].unique()))
        selected_rows = event_data[event_data['condition_id'] == row_id].sort_values('item_id')

    # Extract meta baselines using your updated column names
    first_row = selected_rows.iloc[0]
    ps = float(first_row['ps'])
    dprime_h = float(first_row['dprime_human'])
    dprime_s = float(first_row['dprime_ai'])

    thresholds_distance = str(first_row['thresholds_distance'])
    architecture = str(first_row['architecture'])

    data_dict = {1: {}, 2: {}, 3: {}}
    rows_list = selected_rows.to_dict('records')

    # Distribute the loaded items sequentially across experimental blocks
    # Block 1: Trials 1-10
    for idx, row in enumerate(rows_list[:10]):
        trial_num = idx + 1
        data_dict[1][trial_num] = {
            'event': row['true_label'],
            'stimuli': float(row['x_human']) + STIMULI_SCALAR,
            'ds_judgment': get_ds_state(row['ai_classification'])
        }

    # Block 2: Trials 11-20
    for idx, row in enumerate(rows_list[10:20]):
        trial_num = idx + 1  # Local block indexing (1 to 10)
        data_dict[2][trial_num] = {
            'event': row['true_label'],
            'stimuli': float(row['x_human']) + STIMULI_SCALAR,
            'ds_judgment': get_ds_state(row['ai_classification'])
        }

    # Block 3: Trials 21-120
    for idx, row in enumerate(rows_list[20:120]):
        trial_num = idx + 1  # Local block indexing (1 to 100)
        data_dict[3][trial_num] = {
            'event': row['true_label'],
            'stimuli': float(row['x_human']) + STIMULI_SCALAR,
            'ds_judgment': get_ds_state(row['ai_classification'])
        }

    return data_dict, row_id, ps, dprime_h, dprime_s, thresholds_distance, architecture

def mark_row_in_progress(csv_row_id: int):
    """
    Mark CSV row as used=0.5 when user STARTS experiment.
    NOTE: This is now handled atomically inside load_block_trials() with FileLock.
    This function is kept for backwards compatibility but should not be called for new users.
    """
    if csv_row_id:
        csv_path = os.path.join(settings.BASE_DIR, "data", "data.csv")

        if not os.path.exists(csv_path):
            logger.error(f"CRITICAL: CSV not found at {csv_path} in mark_row_in_progress")
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        try:
            event_data = pd.read_csv(csv_path)
            old_value = event_data.loc[event_data['id'] == csv_row_id, 'used'].values
            old_value = old_value[0] if len(old_value) > 0 else 'NOT_FOUND'

            event_data.loc[event_data['id'] == csv_row_id, 'used'] = 0.5
            event_data.to_csv(csv_path, index=False)

            logger.info(f"Marked row {csv_row_id} as in-progress: {old_value} -> 0.5")
        except Exception as e:
            logger.error(f"Failed to mark row {csv_row_id} as in-progress: {e}")
            raise


def mark_row_as_used(user_id: int):
    """
    Mark CSV row as used=1 when user COMPLETES experiment.
    Called from toast_4 (after questionnaire completion).
    """
    experiment_data = ExperimentData.objects.get(user_id=user_id)
    csv_row_id = experiment_data.csv_row_id

    if csv_row_id:
        csv_path = os.path.join(settings.BASE_DIR, "data", "data.csv")
        lock_path = csv_path + ".lock"

        with FileLock(lock_path, timeout=30):
            event_data = pd.read_csv(csv_path)
            event_data.loc[event_data['id'] == csv_row_id, 'used'] = 1

            # Set isDemo: 1 for old users (demo/pilot), 0 for new users (CloudResearch only)
            # Check if aid is from CloudResearch (not 'test' or local)
            aid = experiment_data.aid
            is_demo = 1 if (aid == 'test' or aid.startswith('local_')) else 0

            if 'isDemo' not in event_data.columns:
                event_data['isDemo'] = None
            event_data.loc[event_data['id'] == csv_row_id, 'isDemo'] = is_demo

            event_data.to_csv(csv_path, index=False)
        logger.info(f"Marked CSV row {csv_row_id} as used=1 (completed)")


def mark_row_as_available(csv_row_id: int):
    """
    Mark CSV row as used=0 when user QUITS/ABANDONS experiment.
    This makes the row available for future users.
    Called when incomplete user is detected.
    """
    if csv_row_id:
        csv_path = os.path.join(settings.BASE_DIR, "data", "data.csv")
        lock_path = csv_path + ".lock"

        with FileLock(lock_path, timeout=30):
            event_data = pd.read_csv(csv_path)
            # Only reset if it's in_progress (0.5), not if already completed (1)
            current_value = event_data.loc[event_data['id'] == csv_row_id, 'used'].values[0]
            if current_value == 0.5:
                event_data.loc[event_data['id'] == csv_row_id, 'used'] = 0
                event_data.to_csv(csv_path, index=False)
                logger.info(f"Reset CSV row {csv_row_id} to used=0 (abandoned)")


def _reset_abandoned_rows():
    """
    Auto-timeout: Reset CSV rows that have been in-progress (used=0.5) for >30 minutes.
    Only checks users with used=0.5 rows (not all incomplete users).
    Called on landing_page() - when user 100 arrives, it checks if user 99's row should be reset.

    OPTIMIZATION: First check without lock, only acquire lock if there's work to do.
    """
    csv_path = os.path.join(settings.BASE_DIR, "data", "data.csv")
    lock_path = csv_path + ".lock"

    # Quick check without lock - just read to see if there's any work to do
    event_data = pd.read_csv(csv_path)
    in_progress_rows = event_data[event_data['used'] == 0.5]

    if len(in_progress_rows) == 0:
        return  # No in-progress rows to check

    # Get the MOST RECENT incomplete user for each csv_row_id
    # This prevents old users from affecting rows assigned to newer users
    csv_row_ids = in_progress_rows['id'].tolist()

    # Find the most recent user for each csv_row_id
    most_recent_users = {}
    for csv_row_id in csv_row_ids:
        latest_user = ExperimentData.objects.filter(
            csv_row_id=csv_row_id
        ).order_by('-start_time').first()

        if latest_user and not latest_user.complete:
            most_recent_users[csv_row_id] = latest_user

    # Determine which rows need to be reset (without modifying anything yet)
    now = datetime.datetime.now()
    timeout_minutes = 30
    rows_to_reset = []

    for csv_row_id, user in most_recent_users.items():
        # csv_row_id already available from the dict key

        # Get last action time (most recent action ID = most recent activity)
        last_action = ExperimentAction.objects.filter(user_id=user.user_id).order_by('-id').first()

        if last_action:
            all_actions = ExperimentAction.objects.filter(user_id=user.user_id)
            total_decision_time = sum(a.decision_time for a in all_actions)

            start_time = user.start_time
            if isinstance(start_time, str):
                start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_time.tzinfo:
                start_time = start_time.replace(tzinfo=None)

            now_naive = now.replace(tzinfo=None) if now.tzinfo else now
            last_activity = start_time + datetime.timedelta(seconds=total_decision_time)
            time_diff = (now_naive - last_activity).total_seconds() / 60
        else:
            start_time = user.start_time
            if isinstance(start_time, str):
                start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_time.tzinfo:
                start_time = start_time.replace(tzinfo=None)

            now_naive = now.replace(tzinfo=None) if now.tzinfo else now
            time_diff = (now_naive - start_time).total_seconds() / 60

        if time_diff > timeout_minutes:
            rows_to_reset.append((user, csv_row_id, last_action))

    # If no rows to reset, we're done
    if len(rows_to_reset) == 0:
        return

    # NOW acquire lock to make changes
    with FileLock(lock_path, timeout=10):
        # Re-read CSV to get fresh state
        event_data = pd.read_csv(csv_path)
        reset_count = 0

        for user, csv_row_id, last_action in rows_to_reset:
            # Verify row is still 0.5 (could have changed while we were processing)
            current_value = event_data.loc[event_data['id'] == csv_row_id, 'used'].values
            if len(current_value) == 0 or current_value[0] != 0.5:
                continue  # Row state changed, skip

            event_data.loc[event_data['id'] == csv_row_id, 'used'] = 0
            reset_count += 1

            # Set end_time
            if not user.end_time:
                if last_action:
                    all_actions = ExperimentAction.objects.filter(user_id=user.user_id).order_by('id')
                    total_decision_time = sum(a.decision_time for a in all_actions)

                    start_time = user.start_time
                    if isinstance(start_time, str):
                        start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    if start_time.tzinfo:
                        start_time = start_time.replace(tzinfo=None)

                    last_action_time = start_time + datetime.timedelta(seconds=total_decision_time)
                    user.end_time = last_action_time.isoformat()
                    user.save()
                else:
                    user.end_time = user.start_time.isoformat() if isinstance(user.start_time, str) else str(user.start_time)
                    user.save()

        # Save if any changes
        if reset_count > 0:
            event_data.to_csv(csv_path, index=False)
            logger.info(f"Reset {reset_count} abandoned rows to used=0")


def landing_page(request):
    # ========== STEP 1: GET AID FROM MULTIPLE POSSIBLE PARAMETERS ==========
    # CloudResearch might use different parameter names
    aid = None
    aid_source = "none"

    # Check multiple possible parameter names (in priority order)
    aid_param_names = ['aid', 'workerId', 'WORKER_ID', 'worker_id', 'participant_id',
                       'participantId', 'session_id', 'sessionId', 'prolific_pid', 'PROLIFIC_PID']

    for param_name in aid_param_names:
        value = request.GET.get(param_name)
        if value and value != '{{WORKER_ID}}' and not value.startswith('{{'):
            aid = value
            aid_source = param_name
            break

    # ========== STEP 2: LOG THIS ATTEMPT IMMEDIATELY (before any DB/CSV ops) ==========
    log_landing_attempt(request, aid if aid else "NO_AID", aid_source)

    # ========== STEP 3: CHECK SESSION FOR EXISTING AID (prevents refresh creating new user) ==========
    if not aid and 'aid' in request.session and 'user_id' in request.session:
        # User refreshed the page - restore their AID from session
        aid = request.session['aid']
        aid_source = "session_restore"
        logger.info(f"Restored AID from session: {aid}")

    # ========== STEP 4: GENERATE TEST AID IF STILL NONE ==========
    if not aid:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        aid = f"test_{timestamp}_{unique_id}"
        logger.info(f"Generated test AID: {aid}")
    else:
        if aid_source != "session_restore":
            logger.info(f"Received AID '{aid}' from parameter '{aid_source}'")

    # ========== STEP 4: AUTO-TIMEOUT FOR ABANDONED ROWS ==========
    try:
        _reset_abandoned_rows()
    except Exception as e:
        logger.error(f"Error in _reset_abandoned_rows: {e}")
    # Check if user already exists (by aid, not just session!)
    try:
        experiment_data = ExperimentData.objects.get(aid=aid)

        # User exists - check if completed
        if experiment_data.complete:
            # If test user already complete, generate new unique aid
            if aid.startswith("test"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:6]
                aid = f"test_{timestamp}_{unique_id}"
                # Raise exception to create new user with new aid
                raise ExperimentData.DoesNotExist
            # Real CloudResearch user who already completed
            return redirect('/end/')

        # Incomplete user - restore their data
        csv_row_id = experiment_data.csv_row_id
        if csv_row_id:
            # Load trials from their assigned row
            events_data, condition_id, ps, dprime_h, dprime_s, thresh_dist, architecture = load_block_trials(csv_row_id=csv_row_id)
        else:
            # Old record without csv_row_id - assign new row
            events_data, condition_id, ps, dprime_h, dprime_s, thresh_dist, architecture = load_block_trials()
            experiment_data.csv_row_id = csv_row_id
            experiment_data.ps = ps
            experiment_data.human_sensitivity = dprime_h
            experiment_data.ds_sensitivity = dprime_s
            experiment_data.thresh_dist = thresh_dist
            experiment_data.architecture = architecture
            experiment_data.save()

        # Restore session
        request.session["user_id"] = experiment_data.user_id
        request.session["aid"] = aid
        request.session["ps"] = float(ps)
        request.session["human_sensitivity"] = float(dprime_h)
        request.session["ds_sensitivity"] = float(dprime_s)
        request.session["thresh_dist"] = thresh_dist
        request.session["architecture"] = architecture
        request.session["events_data"] = events_data
        request.session["csv_row_id"] = csv_row_id
        request.session["block_scores"] = request.session.get("block_scores", {})
        if "experiment_start_time" not in request.session:
            request.session["experiment_start_time"] = datetime.datetime.now().isoformat()

    except ExperimentData.DoesNotExist:
        # New user - assign CSV row
        logger.info(f"Creating new user with AID: {aid}")

        try:
            events_data, csv_row_id, ps, dprime_h, dprime_s = load_block_trials()
            logger.info(f"Assigned CSV row {csv_row_id} to AID {aid}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load_block_trials for AID {aid}: {e}")
            # Log the error to a file for debugging
            error_log_path = os.path.join(settings.BASE_DIR, 'data', 'csv_errors.log')
            with open(error_log_path, 'a') as f:
                f.write(f"{datetime.datetime.now().isoformat()} - load_block_trials failed for {aid}: {e}\n")
            raise  # Re-raise so we can see the error

        # NOTE: Row marking is now done atomically inside load_block_trials() with FileLock
        # The separate mark_row_in_progress() call is no longer needed and is commented out
        # to prevent the race condition that caused duplicate assignments in the first experiment.
        #
        # OLD CODE (caused race condition):
        # try:
        #     mark_row_in_progress(csv_row_id)
        #     logger.info(f"Marked row {csv_row_id} as in-progress (0.5)")
        # except Exception as e:
        #     logger.error(f"CRITICAL: Failed to mark_row_in_progress for row {csv_row_id}, AID {aid}: {e}")
        #     ...
        logger.info(f"Row {csv_row_id} was marked as 0.5 atomically inside load_block_trials()")

        # Create record (use get_or_create to prevent race condition)
        try:
            experiment_data, created = ExperimentData.objects.get_or_create(
                aid=aid,
                defaults={
                    'ps': ps,
                    'human_sensitivity': dprime_h,
                    'ds_sensitivity': dprime_s,
                    'architecture': architecture,
                    'thresh_dist': thresh_dist,
                    'csv_row_id': csv_row_id,
                    'complete': False
                }
            )
            logger.info(f"Created user record: user_id={experiment_data.user_id}, created={created}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create ExperimentData for AID {aid}: {e}")
            error_log_path = os.path.join(settings.BASE_DIR, 'data', 'csv_errors.log')
            with open(error_log_path, 'a') as f:
                f.write(f"{datetime.datetime.now().isoformat()} - ExperimentData creation failed for {aid}: {e}\n")
            raise

        # If not created, someone else created it first - use their data
        if not created:
            # Reset our row back to available since we won't use it
            mark_row_as_available(csv_row_id)
            csv_row_id = experiment_data.csv_row_id
            events_data, condition_id, ps, dprime_h, dprime_s, thresh_dist, architecture = load_block_trials(csv_row_id=csv_row_id)
            logger.info(f"User already existed, restored from row {csv_row_id}")

        # Store in session
        request.session["user_id"] = experiment_data.user_id
        request.session["aid"] = aid
        request.session["ps"] = float(ps)
        request.session["human_sensitivity"] = float(dprime_h)
        request.session["ds_sensitivity"] = float(dprime_s)
        request.session["thresh_dist"] = thresh_dist
        request.session["architecture"] = architecture
        request.session["events_data"] = events_data
        request.session["csv_row_id"] = csv_row_id
        request.session["block_scores"] = {}
        request.session["experiment_start_time"] = datetime.datetime.now().isoformat()

    if request.method == "POST":
        if request.POST['Continue'] == 'continue':
            return redirect('/consent_form/')
    return render(request, 'landing_page.html')


# View for the consent form page
# def consent_form(request):
#     if request.method == "POST":
#         if request.POST['Continue'] == 'begin_experiment':
#             request.session["current_screen"] = 1
#             return redirect('/recaptcha/')
#         elif request.POST['Continue'] == 'end_experiment':
#             return redirect('/end/')  # Redirect to the instruction page (replace with actual URL name)

#     return render(request, 'consent_form.html')

def consent_form(request):
    if request.method == "POST":
        if request.POST['Continue'] == 'begin_experiment':
            request.session["current_screen"] = 1
            return redirect('/recaptcha/')
        elif request.POST['Continue'] == 'end_experiment':
            # If user never started (no user_id), redirect directly to CloudResearch
            # without creating a database entry
            if 'user_id' not in request.session:
                aid = request.session.get("aid", "test")
                return redirect(f'https://app.cloudresearch.com/Router/ThankYouTerm?aid={aid}')
            else:
                # If they started but quit, go to end page to mark as incomplete
                return redirect('/end/')

    return render(request, 'consent_form.html')

def recaptcha(request):
    # Skip reCAPTCHA for local testing
    if request.method == 'POST':
        return redirect('/instructions/')

    # For local testing, just redirect to instructions
    # Uncomment the verification code below for production
    return redirect('/instructions/')

    # PRODUCTION CODE (commented out for local testing):
    # if request.method == 'POST':
    #     response_token = request.POST.get('g-recaptcha-response')
    #     if not response_token:
    #         return render(request, 'form.html', {'error': 'reCAPTCHA not completed.'})
    #
    #     # Verify the token with Google
    #     secret_key = '6LeNJdUrAAAAAFd0vWFtLGbkdxYXQCkM7rfPhnGP'
    #     verify_url = 'https://www.google.com/recaptcha/api/siteverify'
    #     payload = {
    #         'secret': secret_key,
    #         'response': response_token,
    #         'remoteip': request.META.get('REMOTE_ADDR')
    #     }
    #
    #     response = requests.post(verify_url, data=payload)
    #     result = response.json()
    #
    #     if result.get('success'):
    #         return redirect('/instructions/')
    #     else:
    #         return render(request, 'recaptcha.html', {'error': 'Invalid reCAPTCHA. Try again.'})
    # return render(request, 'recaptcha.html')


def instructions(request):
    current_screen = int(request.session.get("current_screen", 1))
    block_scores = request.session.get("block_scores", {})

    def _has_block_score(block_number: int) -> bool:
        return block_number in block_scores or str(block_number) in block_scores

    # Prevent access to Block 2 instructions (screen 4) before completing Block 1
    if current_screen == 4:
        if not _has_block_score(1):
            current_screen = 3
            request.session["current_screen"] = 3

    context = {
        "screen": current_screen,
        'ds_sensitivity': request.session["ds_sensitivity"],
        "v_tp": 1, "v_fp": 1, "v_tn": 1, "v_fn": 2,
    }
    if request.method == "POST":
        if request.POST['Continue'] == 'continue':
            current_screen = int(request.session.get("current_screen", 1))
            if current_screen == 3:
                pass
            else:
                request.session["current_screen"] += 1
        elif request.POST['Continue'] == 'back':
            current_screen = int(request.session.get("current_screen", 1))
            if current_screen == 4 and not _has_block_score(1):
                request.session["current_screen"] = 3
            else:
                request.session["current_screen"] -= 1
        elif request.POST['Continue'] == 'start_block_1':
            request.session["current_screen"] += 1
            request.session["pd"] = False
            request.session["score"] = 30
            request.session["block"] = 1
            request.session["trial"] = 1
            return redirect('/game/')
        elif request.POST['Continue'] == 'start_block_2':
            request.session["current_screen"] += 1
            request.session["pd"] = True
            request.session["score"] = 30
            request.session["block"] = 2
            request.session["trial"] = 1
            return redirect('/game/')
        elif request.POST['Continue'] == 'pd_screen':
            request.session["pd"] = True
            request.session["score"] = 30
            request.session["block"] = 3
            request.session["trial"] = 1
            request.session["default"] = False
            return redirect('/game/')
        return redirect('/instructions/')

    return render(request, "instructions.html", context)


def end(request):
    # If user never started (no user_id), redirect directly to CloudResearch
    if 'user_id' not in request.session:
        aid = request.session.get("aid", "test")
        return redirect(f'https://app.cloudresearch.com/Router/ThankYouTerm?aid={aid}')

    # Check if user completed the experiment:
    # 1. Must have 120 actions (all trials completed)
    # 2. Must have completed TOAST questionnaire
    action_count = ExperimentAction.objects.filter(user_id=request.session["user_id"]).count()
    participant = ExperimentData.objects.get(user_id=request.session["user_id"])

    # Check if TOAST questionnaire was completed
    has_toast_response = TOASTResponse.objects.filter(user_id=request.session["user_id"]).exists()

    # User is complete ONLY if both conditions are met
    if action_count >= 120 and has_toast_response:
        participant.complete = True
        request.session["complete"] = True
    else:
        # Incomplete user - mark as incomplete and reset CSV row to available
        participant.complete = False
        request.session["complete"] = False

        # Set end_time to last action time (if actions exist), otherwise use current time
        last_action = ExperimentAction.objects.filter(user_id=request.session["user_id"]).order_by('-id').first()
        if last_action:
            # Calculate last action time: start_time + sum of all decision_times up to last action
            all_actions = ExperimentAction.objects.filter(user_id=request.session["user_id"]).order_by('id')
            total_decision_time = sum(a.decision_time for a in all_actions)  # seconds

            start_time = participant.start_time
            if isinstance(start_time, str):
                start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_time.tzinfo:
                start_time = start_time.replace(tzinfo=None)

            last_action_time = start_time + datetime.timedelta(seconds=total_decision_time)
            participant.end_time = last_action_time.isoformat()
        else:
            # No actions - use current time
            participant.end_time = datetime.datetime.now().isoformat()

        participant.save()

        # Reset CSV row from 0.5 back to 0 so it can be reused
        if participant.csv_row_id:
            mark_row_as_available(participant.csv_row_id)

        aid = request.session.get("aid", "test")
        return redirect(f'https://app.cloudresearch.com/Router/ThankYouTerm?aid={aid}')

    # Only complete users reach here - update end_time and show completion page
    exp_start_time = datetime.datetime.fromisoformat(request.session["experiment_start_time"])
    exp_end_time = datetime.datetime.now().isoformat()
    participant.end_time = exp_end_time
    participant.save()

    aid = request.session["aid"]

    context = {
        'aid': aid,
        'finish': request.session["complete"],
    }
    return render(request, 'end.html', context)


def game(request):
    """Main loop parsing trial context logs."""
    if request.method == "GET":
        request.session['screen_entry_time'] = datetime.datetime.now().isoformat()

    # Boundary checkers
    if request.session["block"] <= 2 and request.session["trial"] > 10:
        block_scores = request.session.get("block_scores", {})
        if request.session["block"] == 1:
            block_scores["1"] = [request.session["score"], False]
            request.session["current_screen"] = 4
        if request.session["block"] == 2:
            block_scores["2"] = [request.session["score"], True]
            request.session["current_screen"] = 6
        request.session["block_scores"] = block_scores
        return redirect('/instructions/')

    elif request.session["block"] == 3 and request.session["trial"] > 100:
        block_scores = request.session.get("block_scores", {})
        block_scores["3"] = [request.session["score"], request.session["pd"]]
        request.session["block_scores"] = block_scores
        request.session["pd"] = True
        request.session["score"] = 30
        request.session["trial"] = 1
        return redirect('/toast_1/')

    # Grab configuration details
    current_trial_data = request.session["events_data"][str(request.session["block"])][str(request.session["trial"])]
    event_type = current_trial_data['event']
    ds_judgment = current_trial_data['ds_judgment']
    stimuli = round(current_trial_data['stimuli'], 2)

    show_ds = request.session["block"] > 1

    context = {
        'pd': request.session["pd"],
        'event_type': event_type,
        'ds_judgment': ds_judgment,
        'stimuli': stimuli,
        'trial': request.session["trial"],
        'score': request.session["score"],
        'block': request.session["block"],
        'show_ds': show_ds
    }

    if request.method == "POST":
        entry_time = datetime.datetime.fromisoformat(request.session.get('screen_entry_time'))
        time_spent = (datetime.datetime.now() - entry_time).total_seconds()

        user_choice = request.POST['Classification']
        request.session["classification"] = user_choice

        # Scoring logic matrix evaluates choice matching objective ground truth
        if user_choice == 'signal' and event_type == 'signal':
            request.session["score"] += 1
        elif user_choice == 'noise' and event_type == 'noise':
            request.session["score"] += 1
        elif user_choice == 'noise' and event_type == 'signal':
            request.session["score"] -= 1
        elif user_choice == 'signal' and event_type == 'noise':
            request.session["score"] -= 1

        if 'user_id' in request.session:
            experiment_data = ExperimentData.objects.get(user_id=request.session["user_id"])

            # Save historical judgment state directly as string value ('signal' / 'noise' / 'not sure')
            ExperimentAction.objects.update_or_create(
                user_id=experiment_data,
                block_number=request.session["block"],
                trial_number=request.session["trial"],
                defaults={
                    'classification_decision': user_choice,
                    'stimulus_seen': stimuli,
                    'dss_judgment': ds_judgment,
                    'decision_time': time_spent,
                    'correct_classification': event_type
                }
            )

            request.session["trial"] += 1
            del request.session['screen_entry_time']

        return redirect('/game/')

    return render(request, 'game.html', context)


def toast_1(request):
    if request.method == 'POST':
        request.session["q1"] = request.POST.get('usefulness')
        request.session["q2"] = request.POST.get('reliability')
        request.session["q3"] = request.POST.get('trust')
        request.session["q4"] = request.POST.get('confidence')

        return redirect('/toast_2/')

    return render(request, 'toast_1.html')

def toast_2(request):
    if request.method == 'POST':
        request.session["q5"] = request.POST.get('satisfaction')
        request.session["q6"] = request.POST.get('accuracy')
        request.session["q7"] = request.POST.get('consistency')
        request.session["q8"] = request.POST.get('surprised')
        request.session["q9"] = request.POST.get('comfortable')
        return redirect('/toast_3/')

    return render(request, 'toast_2.html')

def toast_3(request):
    if request.method == 'POST':
        request.session["numeracy_fractions"] = request.POST.get('numeracy_fractions')
        request.session["numeracy_shirt"] = request.POST.get('numeracy_shirt')
        request.session["numeracy_useful"] = request.POST.get('numeracy_useful')
        return redirect('/toast_4/')

    return render(request, 'toast_3.html')

def toast_4(request):
    experiment_data = ExperimentData.objects.get(user_id=request.session["user_id"])

    if request.method == 'POST':
        TOASTResponse.objects.create(
            user_id=experiment_data,
            usefulness=request.session["q1"],
            reliability=request.session["q2"],
            trust=request.session["q3"],
            confidence=request.session["q4"],
            satisfaction=request.session["q5"],
            predictability=request.session["q6"],
            understandability=request.session["q7"],
            surprised=request.session["q8"],
            comfortable=request.session["q9"],
            numeracy_fractions=request.session["numeracy_fractions"],
            numeracy_shirt=request.session["numeracy_shirt"],
            numeracy_useful=request.session["numeracy_useful"],
            age_group=request.POST.get('age_group'),
            gender=request.POST.get('gender'),
            education=request.POST.get('education')
        )

        # Mark CSV row as used ONLY when user completes experiment
        mark_row_as_used(experiment_data.user_id)

        return redirect('/end/')

    return render(request, 'toast_4.html')


def save_db(request):
    if request.session.get('authenticated'):
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # ExperimentData export
        users_path = os.path.join(data_dir, 'experiment_data.csv')
        with open(users_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['user_id', 'aid', 'ps', 'human_sensitivity', 'ds_sensitivity',
                             'architecture', 'thresholds_distance',
                             'start_time', 'complete', 'end_time'])
            for user in ExperimentData.objects.order_by('user_id'):
                writer.writerow([
                    user.user_id,
                    user.aid,
                    user.ps,
                    user.human_sensitivity,
                    user.ds_sensitivity,
                    user.architecture,
                    user.thresholds_distance,
                    user.start_time.isoformat() if user.start_time else '',
                    user.complete,
                    user.end_time if user.end_time else ''
                ])

        # ExperimentAction export
        actions_path = os.path.join(data_dir, 'experiment_actions.csv')
        with open(actions_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['user_id', 'block_number', 'trial_number', 'classification_decision',
                             'stimulus_seen', 'dss_judgment', 'decision_time', 'correct_classification'])
            for action in ExperimentAction.objects.order_by('user_id', 'block_number', 'trial_number'):
                writer.writerow([
                    action.user_id.user_id,
                    action.block_number,
                    action.trial_number,
                    action.classification_decision,
                    action.stimulus_seen,
                    action.dss_judgment,
                    action.decision_time,
                    action.correct_classification
                ])

        # TOAST export
        toast_path = os.path.join(data_dir, 'TOAST.csv')
        with open(toast_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['user_id', 'usefulness', 'reliability', 'trust', 'confidence',
                             'satisfaction', 'predictability', 'understandability',
                             'surprised', 'comfortable', 'numeracy_fractions', 'numeracy_shirt',
                             'numeracy_useful', 'age_group', 'gender', 'education'])
            for response in TOASTResponse.objects.order_by('user_id'):
                writer.writerow([
                    response.user_id.user_id,
                    response.usefulness,
                    response.reliability,
                    response.trust,
                    response.confidence,
                    response.satisfaction,
                    response.predictability,
                    response.understandability,
                    response.surprised,
                    response.comfortable,
                    response.numeracy_fractions,
                    response.numeracy_shirt,
                    response.numeracy_useful,
                    response.age_group,
                    response.gender,
                    response.education
                ])

        return redirect('/login/')
    return redirect('/login/')

def login(request):
    if request.method == 'POST':
        if request.POST.get('password') == ADMIN_PASSWORD:
            request.session['authenticated'] = True
            return redirect('progress')
        else:
            return render(request, 'password_prompt.html')
    return render(request, 'password_prompt.html')


def progress(request):
    if request.session['authenticated']:

        users_dict = {}
        for idx, user in enumerate(ExperimentData.objects.all()):
            users_dict[idx] = [user.user_id, user.aid, user.ps, user.human_sensitivity, user.ds_sensitivity, user.start_time,
                               user.complete, user.end_time]

        users_df = pd.DataFrame.from_dict(users_dict, orient='index',
                                          columns=['user_id', 'aid', 'ps', 'human_sensitivity', 'ds_sensitivity', 'start_time',
                                                   'complete', 'end_time'])

        users_df = users_df[users_df['complete'] == True]

        return render(request, 'user_progress.html', {
            'total': users_df.shape[0]
        })
    else:
        return redirect('/login/')

def fresh_restart(request):
    if request.session['authenticated']:
        # Step 1: Clear all Experiment-related data
        ExperimentAction.objects.all().delete()
        ExperimentData.objects.all().delete()

        # Step 2: Clear current user's session
        request.session.flush()

        # Step 3: (Optional) Delete all session records in DB (for all users)
        Session.objects.all().delete()
        return redirect('/login/')
    else:
        return redirect('/login/')


@csrf_exempt
def log_devtools(request):
    """Log when user opens DevTools - writes directly to CSV"""
    if request.method == 'POST':
        user_id = request.session.get('user_id')
        if user_id:
            csv_path = os.path.join(settings.BASE_DIR, 'data', 'devtools_log.csv')
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['user_id', 'details', 'timestamp'])
                writer.writerow([user_id, request.body.decode('utf-8'), datetime.datetime.now().isoformat()])
    return JsonResponse({'status': 'ok'})

