
import cv2
import json
import joblib
import sqlite3
import tempfile
import numpy as np
import mediapipe as mp
import streamlit as st

from pathlib import Path
from streamlit_webrtc import webrtc_streamer

from vision.webcam import DriverProcessor
from agents.decision_agent import DecisionAgent
from agents.alert_agent import AlertAgent


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AEGIS | Driver Safety",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "aegis_incidents.db"
MODEL_PATH = BASE_DIR / "models" / "aegis_vigilance_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "aegis_vigilance_metadata.json"

SAMPLE_VIDEO = Path.home() / "UTA_RLDD_sample" / "sample_video.mp4"

LABELS = {
    "0": "Alert",
    "5": "Low Vigilant",
    "10": "Drowsy",
}

CLASS_ORDER = ["0", "5", "10"]
TEST_FRAMES = 80

mp_face_mesh = mp.solutions.face_mesh


# ==========================================================
# STYLING
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #090F1E;
        color: #E7EDF8;
    }
    [data-testid="stSidebar"] {
        background: #0D1729;
        border-right: 1px solid #24344E;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #F2F6FF;
    }
    [data-testid="stMetric"] {
        background: #111D32;
        border: 1px solid #273954;
        padding: 18px;
        border-radius: 13px;
    }
    [data-testid="stMetricLabel"] {
        color: #9DACC4;
    }
    div.stButton > button {
        border-radius: 9px;
        min-height: 42px;
        border: 1px solid #304564;
    }
    hr {
        border-color: #24344E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metadata():
    if not METADATA_PATH.exists():
        return {}

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


classifier = None
model_error = None

try:
    classifier = load_classifier()
except Exception as error:
    model_error = str(error)

metadata = load_model_metadata()


# ==========================================================
# DATABASE
# ==========================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                ear REAL,
                closed_duration REAL,
                priority TEXT,
                action TEXT,
                message TEXT
            )
            """
        )


def save_incident(monitor, decision):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO incidents (
                timestamp, status, ear, closed_duration,
                priority, action, message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                monitor.get("timestamp", ""),
                monitor.get("status", ""),
                monitor.get("ear"),
                monitor.get("closed_duration"),
                decision.get("priority", ""),
                decision.get("action", ""),
                decision.get("reason", ""),
            ),
        )


def get_incidents():
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM incidents ORDER BY id DESC"
        ).fetchall()


initialise_database()


# ==========================================================
# SESSION STATE
# ==========================================================

if "last_logged_event_id" not in st.session_state:
    st.session_state.last_logged_event_id = None

if "last_alerted_event_id" not in st.session_state:
    st.session_state.last_alerted_event_id = None

if "alert_agent" not in st.session_state:
    st.session_state.alert_agent = AlertAgent(
        cooldown_seconds=8,
        alarm_duration=6,
    )


# ==========================================================
# FEATURE EXTRACTION FOR DATASET TESTING
# Must match train_classifier.py
# ==========================================================

def distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def extract_frame_features(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    height, width = frame.shape[:2]

    def point(index):
        return (
            landmarks[index].x * width,
            landmarks[index].y * height,
        )

    def calculate_ear(indices):
        points = [point(index) for index in indices]
        horizontal = distance(points[0], points[3])

        if horizontal < 1e-6:
            return np.nan

        return (
            distance(points[1], points[5])
            + distance(points[2], points[4])
        ) / (2.0 * horizontal)

    left_ear = calculate_ear([33, 160, 158, 133, 153, 144])
    right_ear = calculate_ear([362, 385, 387, 263, 373, 380])
    ear_value = (left_ear + right_ear) / 2.0

    mouth_vertical = distance(point(13), point(14))
    mouth_horizontal = distance(point(78), point(308))

    mar_value = (
        mouth_vertical / mouth_horizontal
        if mouth_horizontal > 1e-6
        else np.nan
    )

    face_width = distance(point(234), point(454)) / width
    face_height = distance(point(10), point(152)) / height

    features = np.asarray(
        [ear_value, mar_value, face_width, face_height],
        dtype=float,
    )

    if not np.all(np.isfinite(features)):
        return None

    return features


def analyse_video(video_path, progress_bar=None):
    if classifier is None:
        raise RuntimeError("The trained classifier could not be loaded.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("The video contains no readable frames.")

    indices = np.linspace(
        0,
        total_frames - 1,
        min(TEST_FRAMES, total_frames),
        dtype=int,
    )

    feature_rows = []
    frame_numbers = []

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            for position, frame_index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = cap.read()

                if success:
                    features = extract_frame_features(frame, face_mesh)

                    if features is not None:
                        feature_rows.append(features)
                        frame_numbers.append(int(frame_index))

                if progress_bar is not None:
                    progress_bar.progress(
                        (position + 1) / len(indices),
                        text=f"Analysing sample {position + 1}/{len(indices)}",
                    )

    finally:
        cap.release()

    if not feature_rows:
        raise RuntimeError(
            "No usable face frames were detected. Try a clearer video "
            "with the driver's face visible."
        )

    X_test = np.asarray(feature_rows, dtype=float)
    predictions = classifier.predict(X_test)

    results = []

    for frame_number, features, prediction in zip(
        frame_numbers, X_test, predictions
    ):
        results.append(
            {
                "Frame": frame_number,
                "Prediction": LABELS.get(str(prediction), str(prediction)),
                "EAR": round(float(features[0]), 4),
                "MAR": round(float(features[1]), 4),
            }
        )

    counts = {
        LABELS[code]: int(
            sum(str(prediction) == code for prediction in predictions)
        )
        for code in CLASS_ORDER
    }

    return results, counts, len(indices), len(frame_numbers)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("# ◈ AEGIS")
    st.caption("AGENTIC DRIVER SAFETY")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Monitoring",
            "Dataset Testing",
            "Agent Control Centre",
            "Incident History",
            "Model Evaluation",
        ],
    )

    st.divider()
    st.caption("LOCAL PROOF OF CONCEPT")
    st.caption("Camera processing runs locally.")
    st.caption("Alarm: maximum 6 seconds")

    if classifier is not None:
        st.success("Dataset classifier loaded")
    else:
        st.warning("Dataset classifier unavailable")


# ==========================================================
# HEADER
# ==========================================================

st.markdown("## ◈ AEGIS")
st.caption(
    "Intelligent Driver Monitoring · "
    "Computer Vision · Agentic Safety Workflow"
)
st.divider()


# ==========================================================
# OVERVIEW
# ==========================================================

if page == "Overview":

    st.subheader("Safety Operations")
    incidents = get_incidents()

    sustained_events = sum(
        1 for item in incidents
        if item["status"] == "SUSTAINED_EYE_CLOSURE"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Recorded incidents", len(incidents))
    col2.metric("Prolonged-closure events", sustained_events)
    col3.metric("Agent modules", "3")

    st.caption(
        "Prolonged eye closure is an observation, not proof of drowsiness."
    )

    st.divider()
    st.subheader("System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### Vision Engine")
            st.write("Webcam → Facial landmarks → EAR / MAR")
            st.success("Available when camera monitoring is active.")

    with col2:
        with st.container(border=True):
            st.markdown("#### Agent Workflow")
            st.write("Monitor → Decision → Alert")
            st.success("Monitor, decision, and alert modules are connected.")

    st.divider()
    st.subheader("Dataset Classifier")

    if classifier is not None:
        st.success("The trained experimental classifier is loaded.")
        st.caption(
            "Open Dataset Testing to analyse a prerecorded video."
        )
    else:
        st.warning(
            "No trained model is available. Run train_classifier.py first."
        )

    st.divider()
    st.subheader("Recent Incidents")

    if incidents:
        st.dataframe(
            [
                {
                    "Time": row["timestamp"],
                    "Status": row["status"],
                    "Priority": row["priority"],
                    "Action": row["action"],
                }
                for row in incidents[:5]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No incidents recorded yet. Start Live Monitoring.")


# ==========================================================
# LIVE MONITORING
# Existing webcam alarm logic preserved
# ==========================================================

elif page == "Live Monitoring":

    st.subheader("Live Driver Monitoring")
    st.caption("Live webcam input with agent results.")

    st.warning(
        "This prototype detects possible prolonged eye closure. "
        "It is not a validated drowsiness classifier or vehicle safety system."
    )

    camera_context = webrtc_streamer(
        key="aegis_driver_camera",
        video_processor_factory=DriverProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()
    st.subheader("Live Agent Results")

    @st.fragment(run_every=1)
    def live_agent_panel():

        processor = getattr(camera_context, "video_processor", None)

        if processor is None:
            st.info("Start the camera to see live agent results.")
            return

        monitor = getattr(processor, "latest_event", None)

        if not monitor:
            st.info("Waiting for the first webcam observation...")
            return

        decision = DecisionAgent().decide(monitor)

        status = monitor.get("status", "UNABLE_TO_ASSESS")
        action = decision.get("action", "WAIT")
        priority = decision.get("priority", "LOW")
        event_id = monitor.get("event_id")

        alert_agent = st.session_state.alert_agent
        alert_result = None

        # Stop the alarm when sustained eye closure is no longer reported.
        if status != "SUSTAINED_EYE_CLOSURE":
            if alert_agent.is_active():
                alert_agent.stop()

        # Trigger the alarm once for each new closure event.
        elif (
            event_id is not None
            and event_id != st.session_state.last_alerted_event_id
        ):
            st.session_state.last_alerted_event_id = event_id
            alert_result = alert_agent.execute(
                {"alert_required": True}
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Monitor status", status.replace("_", " ").title())
        col2.metric("Decision action", action.replace("_", " "))
        col3.metric("Priority", priority)

        ear = monitor.get("ear")
        duration = monitor.get("closed_duration", 0.0) or 0.0

        col1, col2 = st.columns(2)
        col1.metric(
            "Eye Aspect Ratio (EAR)",
            f"{ear:.3f}" if ear is not None else "N/A",
        )
        col2.metric("Eye-closure duration", f"{duration:.1f} s")

        st.write("**Monitor Agent:**", monitor.get("message", ""))
        st.write("**Decision Agent:**", decision.get("reason", ""))

        if status == "SUSTAINED_EYE_CLOSURE":
            if (
                event_id is not None
                and event_id != st.session_state.last_logged_event_id
            ):
                save_incident(monitor, decision)
                st.session_state.last_logged_event_id = event_id

            st.error(
                "Possible prolonged eye closure detected. "
                "Further assessment is required."
            )
        else:
            st.info("No sustained eye-closure event currently detected.")

        if alert_agent.is_active():
            st.error("🔊 WARNING ALARM ACTIVE")

            if st.button("Stop alarm now", key="stop_aegis_alarm"):
                alert_agent.stop()
                st.success("Alarm stopped.")
        elif status == "SUSTAINED_EYE_CLOSURE":
            st.warning(
                "Sustained eye closure detected. Check the alert message below."
            )
        else:
            st.caption(
                "Alarm is on standby. Normal blinks should not trigger it."
            )

        if alert_result is not None:
            st.caption(alert_result.get("message", ""))

        st.caption(
            "The alarm is designed to stop when the eyes reopen "
            "or after its maximum duration of 6 seconds."
        )
        st.caption("Existing historical records are not deleted.")

    live_agent_panel()


# ==========================================================
# DATASET TESTING
# Uses the saved trained classifier
# ==========================================================

elif page == "Dataset Testing":

    st.subheader("UTA-RLDD Dataset Testing")

    st.caption(
        "Upload a prerecorded video to inspect sampled frames "
        "and run the experimental trained classifier."
    )

    st.warning(
        "The classifier achieved low subject-wise accuracy in the "
        "current evaluation. Predictions are experimental and must not "
        "be used as real-world driving safety decisions."
    )

    uploaded_video = st.file_uploader(
        "Upload a dataset video",
        type=["mp4", "mov", "avi", "mkv"],
        key="dataset_video_upload",
    )

    video_path = None
    is_temporary = False

    if uploaded_video is not None:
        suffix = Path(uploaded_video.name).suffix or ".mp4"

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                video_path = Path(temp_file.name)
                is_temporary = True

        except OSError as error:
            st.error(f"Could not prepare uploaded video: {error}")

    elif SAMPLE_VIDEO.exists():
        video_path = SAMPLE_VIDEO

    if video_path is None:
        st.info(
            "Upload a video to begin. The default sample video "
            "was not found."
        )
        st.caption(f"Default video location: {SAMPLE_VIDEO}")

    else:
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                st.error("OpenCV could not open this video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total frames", f"{frame_count:,}")
                col2.metric(
                    "Frame rate",
                    f"{fps:.2f} FPS" if fps > 0 else "Unknown",
                )
                col3.metric("Duration", f"{duration / 60:.1f} min")
                col4.metric("Resolution", f"{width} × {height}")

                cap.release()

                st.markdown("### Video Preview")
                st.video(str(video_path))

                st.divider()
                st.markdown("### Frame Inspection")

                if frame_count > 0:
                    frame_index = st.slider(
                        "Select a frame",
                        min_value=0,
                        max_value=frame_count - 1,
                        value=0,
                        key="dataset_frame_index",
                    )

                    preview_cap = cv2.VideoCapture(str(video_path))
                    preview_cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(frame_index),
                    )
                    success, frame = preview_cap.read()
                    preview_cap.release()

                    if success:
                        st.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_index}",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not decode the selected frame.")

                st.divider()
                st.markdown("### Trained Classifier Results")

                if classifier is None:
                    st.error(
                        "The trained model could not be loaded. "
                        "Check the models folder and the displayed error."
                    )
                    if model_error:
                        st.code(model_error)
                else:
                    st.caption(
                        "The model samples up to 80 frames evenly across "
                        "the video and predicts each detected face frame."
                    )

                    if st.button(
                        "Analyse video with trained model",
                        type="primary",
                        key="analyse_dataset_video",
                    ):
                        progress = st.progress(
                            0.0,
                            text="Preparing video analysis...",
                        )

                        try:
                            results, counts, sampled, detected = analyse_video(
                                video_path,
                                progress,
                            )

                            progress.empty()

                            st.session_state.dataset_results = results
                            st.session_state.dataset_counts = counts
                            st.session_state.dataset_sampled = sampled
                            st.session_state.dataset_detected = detected
                            st.session_state.dataset_filename = (
                                uploaded_video.name
                                if uploaded_video is not None
                                else video_path.name
                            )

                        except Exception as error:
                            progress.empty()
                            st.error(f"Video analysis failed: {error}")

                if "dataset_results" in st.session_state:
                    results = st.session_state.dataset_results
                    counts = st.session_state.dataset_counts
                    sampled = st.session_state.dataset_sampled
                    detected = st.session_state.dataset_detected

                    st.markdown("#### Prediction Summary")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frames sampled", sampled)
                    col2.metric("Frames with detected face", detected)
                    col3.metric(
                        "Frames analysed",
                        len(results),
                    )

                    prediction_total = len(results)

                    if prediction_total:
                        for class_name in [
                            "Alert",
                            "Low Vigilant",
                            "Drowsy",
                        ]:
                            count = counts[class_name]
                            percentage = 100 * count / prediction_total

                            st.write(
                                f"**{class_name}:** {count} frames "
                                f"({percentage:.1f}%)"
                            )
                            st.progress(percentage / 100)

                        most_common = max(counts, key=counts.get)

                        st.info(
                            f"Most frequent sampled-frame prediction: "
                            f"**{most_common}**. This is an experimental "
                            f"summary, not a validated overall vigilance result."
                        )

                        st.markdown("#### Frame-by-frame Predictions")

                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True,
                        )

                        csv_data = __import__("pandas").DataFrame(
                            results
                        ).to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Download prediction results (CSV)",
                            data=csv_data,
                            file_name="aegis_dataset_predictions.csv",
                            mime="text/csv",
                        )

                    else:
                        st.warning(
                            "No face frames were available for prediction."
                        )

        finally:
            # The video has already been processed and rendered.
            # Uploaded files are temporary copies, not source dataset files.
            if is_temporary and video_path is not None:
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    pass


# ==========================================================
# AGENT CONTROL CENTRE
# ==========================================================

elif page == "Agent Control Centre":

    st.subheader("Agent Control Centre")
    st.caption("Implemented modules and current integration status.")

    agent_details = [
        (
            "Monitor Agent",
            "agents/monitor_agent.py",
            "Analyses eye measurements and face availability.",
            "Implemented",
        ),
        (
            "Decision Agent",
            "agents/decision_agent.py",
            "Chooses actions from monitoring results.",
            "Implemented",
        ),
        (
            "Agent Orchestrator",
            "agents/orchestrator.py",
            "Coordinates monitoring and decision modules.",
            "Independently tested; live integration pending",
        ),
        (
            "Alert Agent",
            "agents/alert_agent.py",
            "Plays a warning tone for up to 6 seconds.",
            "Connected to live monitoring",
        ),
        (
            "Dataset Classifier",
            "models/aegis_vigilance_model.joblib",
            "Experimental Random Forest using four facial measurements.",
            "Loaded" if classifier is not None else "Unavailable",
        ),
        (
            "Incident Database",
            "aegis_incidents.db",
            "Stores prolonged-eye-closure events.",
            "Connected",
        ),
    ]

    for name, file_name, description, status_text in agent_details:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{name}**")
                st.caption(file_name)
                st.write(description)

            with col2:
                st.write("**Status**")
                st.write(status_text)


# ==========================================================
# INCIDENT HISTORY
# ==========================================================

elif page == "Incident History":

    st.subheader("Incident History")
    incidents = get_incidents()

    if incidents:
        st.metric("Total recorded incidents", len(incidents))

        st.dataframe(
            [
                {
                    "ID": row["id"],
                    "Timestamp": row["timestamp"],
                    "Status": row["status"],
                    "EAR": row["ear"],
                    "Closure (s)": row["closed_duration"],
                    "Priority": row["priority"],
                    "Action": row["action"],
                    "Message": row["message"],
                }
                for row in incidents
            ],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Refresh incident history"):
            st.rerun()
    else:
        st.info("No incidents recorded yet.")


# ==========================================================
# MODEL EVALUATION
# ==========================================================

elif page == "Model Evaluation":

    st.subheader("Model Evaluation")

    if classifier is None:
        st.error("The trained classifier could not be loaded.")

        if model_error:
            st.code(model_error)

        st.info(
            "Check that models/aegis_vigilance_model.joblib exists "
            "and run train_classifier.py if needed."
        )

    else:
        st.success("Trained Random Forest model loaded successfully.")

        col1, col2, col3 = st.columns(3)

        col1.metric("Algorithm", "Random Forest")
        col2.metric("Input features", "4")
        col3.metric("Classes", "3")

        accuracy = metadata.get(
            "subject_wise_mean_accuracy",
            metadata.get("mean_subject_wise_accuracy"),
        )

        if accuracy is not None:
            st.metric(
                "Mean subject-wise accuracy",
                f"{float(accuracy) * 100:.2f}%",
            )

        st.markdown("### Training Details")

        st.write(
            "**Evaluation method:** Leave-one-subject-out cross-validation"
        )

        st.write(
            "**Features:** Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), "
            "face-width ratio, and face-height ratio."
        )

        subjects = metadata.get("training_subjects", [])
        if subjects:
            st.write("**Subjects used for final training:**", ", ".join(subjects))

        if metadata.get("usable_video_count") is not None:
            st.write(
                "**Usable videos:**",
                metadata["usable_video_count"],
            )
        elif metadata.get("video_count") is not None:
            st.write("**Videos discovered:**", metadata["video_count"])

        if metadata.get("frame_count") is not None:
            st.write("**Sampled face frames:**", metadata["frame_count"])

        st.markdown("### Evaluation Interpretation")

        st.warning(
            "The latest reported mean subject-wise accuracy is approximately "
            "26.75%. This is a preliminary result and is too low for dependable "
            "driver-safety classification."
        )

        st.write(
            "The model's predictions are based on four facial measurements "
            "from individual frames. It does not yet model eye-closure "
            "duration or temporal patterns across a video."
        )

        st.caption(
            "The accuracy shown above comes from the saved training metadata. "
            "Class-wise precision, recall, and F1 scores are not stored in "
            "that metadata file, so they are not reconstructed here."
        )

        st.markdown("### Class Labels")

        st.dataframe(
            [
                {"Label code": "0", "Class": "Alert"},
                {"Label code": "5", "Class": "Low Vigilant"},
                {"Label code": "10", "Class": "Drowsy"},
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "The dataset classifier is separate from the live webcam alarm. "
            "Its predictions do not control or replace the existing alarm."
        )
