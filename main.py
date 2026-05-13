import streamlit as st
import os
import time
from urllib import parse, request
from dotenv import load_dotenv
import pandas as pd
from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles
from services.persistence.exercise_repository import init_db
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from services.persistence.exercise_repository import get_users_exercises
from groq import Groq
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio


DEFAULT_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun2.l.google.com:19302"]},
    {"urls": ["stun:stun3.l.google.com:19302"]},
    {"urls": ["stun:stun4.l.google.com:19302"]},
]


def _get_config_value(name, default="", section="webrtc"):
    value = os.environ.get(name)

    if value:
        return value

    try:
        if name in st.secrets:
            return st.secrets[name]

        section_secrets = st.secrets.get(section, {})
        if name in section_secrets:
            return section_secrets[name]
    except Exception:
        return default

    return default


def _as_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, (list, tuple)):
        return [item for item in value if item]

    return []


def _is_openrelay_url(turn_url):
    return isinstance(turn_url, str) and "openrelay.metered.ca" in turn_url


@st.cache_data(ttl=3300, show_spinner=False)
def _fetch_twilio_ice_servers(account_sid, auth_token, ttl=3600):
    token_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
    data = parse.urlencode({"Ttl": ttl}).encode("utf-8")
    token_request = request.Request(token_url, data=data, method="POST")

    password_manager = request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, token_url, account_sid, auth_token)
    opener = request.build_opener(request.HTTPBasicAuthHandler(password_manager))

    with opener.open(token_request, timeout=10) as response:
        payload = response.read().decode("utf-8")

    import json

    return json.loads(payload).get("ice_servers", [])


def _get_twilio_ice_servers():
    account_sid = _get_config_value("TWILIO_ACCOUNT_SID", section="twilio")
    auth_token = _get_config_value("TWILIO_AUTH_TOKEN", section="twilio")

    if not account_sid or not auth_token:
        return []

    try:
        return _fetch_twilio_ice_servers(account_sid, auth_token)
    except Exception:
        return []


def _get_static_turn_servers():
    turn_urls = _as_list(_get_config_value("TURN_URLS"))

    if not turn_urls:
        return []

    turn_username = _get_config_value("TURN_USERNAME")
    turn_credential = _get_config_value("TURN_CREDENTIAL")
    turn_servers = []

    for turn_url in turn_urls:
        if _is_openrelay_url(turn_url):
            continue

        turn_server = {"urls": turn_url}

        if turn_username and turn_credential:
            turn_server["username"] = turn_username
            turn_server["credential"] = turn_credential

        turn_servers.append(turn_server)

    return turn_servers


def get_rtc_configuration():
    ice_servers = _get_twilio_ice_servers()

    if not ice_servers:
        ice_servers = _get_static_turn_servers()

    ice_servers = [*ice_servers, *DEFAULT_ICE_SERVERS]

    return {"iceServers": ice_servers}


def get_server_rtc_configuration():
    return {"iceServers": DEFAULT_ICE_SERVERS}


def get_frontend_rtc_configuration():
    rtc_configuration = get_rtc_configuration()

    if has_turn_server(rtc_configuration):
        rtc_configuration["iceTransportPolicy"] = "relay"

    return rtc_configuration


def has_turn_server(rtc_configuration):
    ice_servers = rtc_configuration.get("iceServers", [])

    for ice_server in ice_servers:
        urls = _as_list(ice_server.get("urls"))
        if any(url.startswith("turn:") or url.startswith("turns:") for url in urls):
            return True

    return False


@st.fragment(run_every=1)
def render_live_sidebar_metrics():
    context = st.session_state.get("webrtc_context")
    sync_metrics_update(context)

    if not st.session_state.get("workout_started", False):
        return

    exercise = st.session_state.get("exercise_type")
    total_reps = st.session_state.get("reps")
    current_set_reps = st.session_state.get("current_set_reps")
    reps_per_set = st.session_state.get("reps_per_set")
    sets_completed = st.session_state.get("sets_completed")
    target_sets = st.session_state.get("target_sets")

    st.divider()
    st.subheader("Progress")
    st.metric("Total Reps", f"{total_reps}")
    st.metric("Current Set Reps", f"{current_set_reps} / {reps_per_set}")
    st.metric("Sets Completed", f"{sets_completed} / {target_sets}")
    st.divider()

    if exercise == "Squats":
        st.subheader("Squat Metrics")
        st.metric("Knee Angle", f"{st.session_state.knee_angle}°")
        st.metric("Back Angle", f"{st.session_state.back_angle}°")
        st.metric("Depth Status", st.session_state.depth_status)

    elif exercise == "Push-ups":
        st.subheader("Push-up Metrics")
        st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
        st.metric("Body Alignment", st.session_state.body_alignment)
        st.metric("Hip Position", st.session_state.hip_status)

    elif exercise == "Biceps Curls (Dumbbell)":
        st.subheader("Curl Metrics")
        st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
        st.metric("Shoulder Stability", st.session_state.shoulder_status)
        st.metric("Swing Detection", st.session_state.swing_status)

    elif exercise == "Shoulder Press":
        st.subheader("Shoulder Press Metrics")
        st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
        st.metric("Arm Extension", st.session_state.extension_status)
        st.metric("Back Arch", st.session_state.back_arch_status)

    elif exercise == "Lunges":
        st.subheader("Lunge Metrics")
        st.metric("Front Knee Angle", f"{st.session_state.front_knee_angle}°")
        st.metric("Torso Angle", f"{st.session_state.torso_angle}°")
        st.metric("Balance Status", st.session_state.balance_status)


@st.fragment(run_every=1)
def render_live_coach_feedback():
    audio_to_play = st.session_state.pop("audio_to_play", None)
    coach_feedback = st.session_state.get("coach_feedback")

    if audio_to_play:
        autoplay_audio(audio_to_play)

    if coach_feedback:
        st.markdown("")
        st.success(f"🤖 **Coach:** {coach_feedback}")

  
def main():
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered"
    )
    
    load_dotenv()

    load_css(os.path.join(os.getcwd(), "static", "style.css"))
    inject_local_font(os.path.join(os.getcwd(), "static", "AdobeClean.otf"), "AdobeClean")

    init_db()

    if not render_login_wall():
        return 

    initial_session_defaults()


    if "voice_pipeline" not in st.session_state:
        try:
            api_key = _get_config_value("GROQ_API_KEY", section="groq")

            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured.")
            
            groq_client = Groq(api_key=api_key)
            llm_coach = LLMCoach(groq_client)
            tts = TextToSpeech()
            st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)
        except Exception as e:
            st.warning(f"Voice coaching is disabled: {e}")
            st.session_state.voice_pipeline = None

    workout_started = st.session_state.get("workout_started", False)
    
    with st.sidebar:
        st.title("🏋️‍♂️ Apna AI Coach")

        if st.session_state.username:
            st.caption(f"👤 Login as {st.session_state.username}")

        st.divider()

        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options=EXERCISE_OPTIONS, key="plan_exercise")

            plan_sets = st.number_input("Sets", min_value=0, max_value=50, key="plan_sets", step=1)

            plan_reps = st.number_input("Reps per Set", min_value=0, max_value=50, key="plan_reps", step=1)

            st.markdown("")

            start_session_button = st.button("Start Workout", width="stretch", key="start_session_button")

            if start_session_button:
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.workout_started = True
                st.session_state.set_cycle_started_at = time.time()
                st.session_state.last_saved_sets_completed = 0

                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_started",
                        exercise=plan_exercise,
                        metrics={}
                    )
                    
                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result

                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False
                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")

            st.info(f"**{exercise}** -- {sets} Sets / {reps} Reps")

            end_session_button = st.button("End Workout", key="end_session_button", width="stretch")

            if end_session_button:
                st.session_state.workout_started = False
                
                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_completed",
                        exercise=exercise,
                        metrics={}
                    )
                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result

                st.rerun()

        render_live_sidebar_metrics()

        if False and workout_started:
            st.divider()

            exercise = st.session_state.get("exercise_type")
            total_reps = st.session_state.get("reps")
            current_set_reps = st.session_state.get("current_set_reps")
            reps_per_set = st.session_state.get("reps_per_set")
            sets_completed = st.session_state.get("sets_completed")
            target_sets = st.session_state.get("target_sets")

            st.subheader("Progress")

            st.metric("Total Reps", f"{total_reps}")
            st.metric("Current Set Reps", f"{current_set_reps} / {reps_per_set}")
            st.metric("Sets Completed", f"{sets_completed} / {target_sets}")

            st.divider()

            if exercise == "Squats":
                st.subheader("Squat Metrics")
                st.metric("Knee Angle", f"{st.session_state.knee_angle}°")
                st.metric("Back Angle", f"{st.session_state.back_angle}°")
                st.metric("Depth Status", st.session_state.depth_status)

            elif exercise == "Push-ups":
                st.subheader("Push-up Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Body Alignment", st.session_state.body_alignment)
                st.metric("Hip Position", st.session_state.hip_status)

            elif exercise == "Biceps Curls (Dumbbell)":
                st.subheader("Curl Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Shoulder Stability", st.session_state.shoulder_status)
                st.metric("Swing Detection", st.session_state.swing_status)

            elif exercise == "Shoulder Press":
                st.subheader("Shoulder Press Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Arm Extension", st.session_state.extension_status)
                st.metric("Back Arch", st.session_state.back_arch_status)

            elif exercise == "Lunges":
                st.subheader("Lunge Metrics")
                st.metric("Front Knee Angle", f"{st.session_state.front_knee_angle}°")
                st.metric("Torso Angle", f"{st.session_state.torso_angle}°")
                st.metric("Balance Status", st.session_state.balance_status)

    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")
    render_live_coach_feedback()
 
    if False and st.session_state.get("audio_to_play"):
        autoplay_audio(st.session_state.audio_to_play)

    if False and st.session_state.get("coach_feedback"):
        st.markdown("")
        st.success(f"🤖 **Coach:** {st.session_state.coach_feedback}")

    if not workout_started:
        st.markdown(
            """
            <div style="
                border: 10px dashed #444;
                border-radius: 0px;
                padding: 48px 32px;
                text-align: center;
                color: #888;
                margin-top: 32px;
                margin-bottom: 32px;
            ">
                <h2 style="color:#ccc; margin-bottom:8px;">👈 Set your workout plan</h2>
                <p style="font-size:1.05rem;">
                    Choose your exercise, sets and reps in the sidebar,<br>
                    then click <strong>Start Workout</strong> to activate the camera and AI coach.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        frontend_rtc_configuration = get_frontend_rtc_configuration()
        server_rtc_configuration = get_server_rtc_configuration()

        if not has_turn_server(frontend_rtc_configuration):
            st.warning(
                "A reliable TURN server is not configured. Add Twilio credentials "
                "to Streamlit Secrets for stable camera streaming on Streamlit Cloud."
            )

        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            frontend_rtc_configuration=frontend_rtc_configuration,
            server_rtc_configuration=server_rtc_configuration,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.session_state.webrtc_context = context
        sync_metrics_update(context)

        inject_webrtc_styles()

    st.divider()

    st.markdown("#### Workout History")

    user_id = st.session_state.get("user_id", 0)

    if isinstance(user_id, int):
        history_rows = get_users_exercises(user_id)

        arr = [
            {
                "Exercise": row['exercise_name'],
                "Reps": row['reps'],
                "Sets": row['sets'],
                "Time (sec)": row['time'],
                "Date": row['created_at']
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            agg_df = df.groupby(["Exercise", "Date"]).agg({
                "Reps": 'sum',
                "Sets": "sum",
                "Time (sec)": "sum"
            }).reset_index()
            agg_df.index += 1
            st.table(agg_df, border="horizontal")
        else:
            st.info("No workout history found.")


if __name__ == "__main__":
    main()
    
