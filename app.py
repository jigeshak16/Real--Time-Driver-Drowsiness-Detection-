import tensorflow as tf
import cv2
import json
import sqlite3
import tempfile
import numpy as np
import mediapipe as mp
import streamlit as st
import pandas as pd

from pathlib import Path
from streamlit_webrtc import webrtc_streamer

from vision.webcam import DriverProcessor
from agents.decision_agent import DecisionAgent
from agents.alert_agent import AlertAgent


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AEGIS | Driver Safety",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "aegis_incidents.db"
MODEL_PATH = BASE_DIR / "models" / "aegis_cnn_lstm_attention.keras"
METADATA_PATH = BASE_DIR / "models" / "aegis_cnn_lstm_metadata.json"

CLASS_NAMES = ["Alert", "Low Vigilant", "Drowsy"]
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


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
# MODEL LOADING — TENSORFLOW LAMBDA FIX
# ==========================================================

def expose_tensorflow_to_lambda_layers(model):
    """
    Expose TensorFlow in the global namespace of deserialized
    Lambda functions that refer to tf or tensorflow.
    """

    visited = set()

    def fix_layers(current_model):
        if id(current_model) in visited:
            return

        visited.add(id(current_model))

        for layer in getattr(current_model, "layers", []):
            if isinstance(layer, tf.keras.layers.Lambda):
                function = getattr(layer, "function", None)

                function_globals = getattr(
                    function,
                    "__globals__",
                    None,
                )

                if isinstance(function_globals, dict):
                    function_globals["tf"] = tf
                    function_globals["tensorflow"] = tf

            # Handle nested models and layer containers.
            nested_layers = getattr(layer, "layers", None)

            if nested_layers is not None:
                fix_layers(layer)

    fix_layers(model)
    return model


@st.cache_resource
def load_classifier():
    """Load the trained CNN-LSTM attention model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN-LSTM model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={
            "tf": tf,
            "tensorflow": tf,
        },
    )

    model = expose_tensorflow_to_lambda_layers(model)

    return model


@st.cache_data
def load_model_metadata():
    if not METADATA_PATH.exists():
        return {}

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError):
        return {}


classifier = None
model_error = None

try:
    classifier = load_classifier()

except Exception as error:
    model_error = f"{type(error).__name__}: {error}"

metadata = load_model_metadata()


# ==========================================================
# DATABASE
# ==========================================================

def get_connection():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )
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
                timestamp,
                status,
                ear,
                closed_duration,
                priority,
                action,
                message
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
        rows = connection.execute(
            "SELECT * FROM incidents ORDER BY id DESC"
        ).fetchall()

    return [dict(row) for row in rows]


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
# FACIAL MEASUREMENT HELPERS
# ==========================================================

def distance(a, b):
    return np.linalg.norm(
        np.asarray(a) - np.asarray(b)
    )


def extract_frame_features(frame, face_mesh):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

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

        horizontal = distance(
            points[0],
            points[3],
        )

        if horizontal < 1e-6:
            return np.nan

        return (
            distance(points[1], points[5])
            + distance(points[2], points[4])
        ) / (2.0 * horizontal)

    left_ear = calculate_ear(
        [33, 160, 158, 133, 153, 144]
    )

    right_ear = calculate_ear(
        [362, 385, 387, 263, 373, 380]
    )

    ear_value = (left_ear + right_ear) / 2.0

    mouth_vertical = distance(
        point(13),
        point(14),
    )

    mouth_horizontal = distance(
        point(78),
        point(308),
    )

    mar_value = (
        mouth_vertical / mouth_horizontal
        if mouth_horizontal > 1e-6
        else np.nan
    )

    face_width = (
        distance(point(234), point(454)) / width
    )

    face_height = (
        distance(point(10), point(152)) / height
    )

    features = np.asarray(
        [
            ear_value,
            mar_value,
            face_width,
            face_height,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(features)):
        return None

    return features


# ==========================================================
# CNN-LSTM VIDEO ANALYSIS
# ==========================================================

def analyse_video(video_path, progress_bar=None):

    if classifier is None:
        raise RuntimeError(
            "The CNN-LSTM model is unavailable. "
            f"Model loading error: {model_error}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "OpenCV could not open this video."
        )

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise RuntimeError(
                "The video contains no readable frames."
            )

        indices = np.linspace(
            0,
            total_frames - 1,
            min(SEQUENCE_LENGTH, total_frames),
            dtype=int,
        )

        frames = []

        for position, frame_index in enumerate(indices):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame = cv2.resize(
                frame,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
            )

            frames.append(frame)

            if progress_bar is not None:
                progress_bar.progress(
                    (position + 1) / len(indices),
                    text=(
                        f"Preparing frame "
                        f"{position + 1}/{len(indices)}"
                    ),
                )

    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            "No frames could be decoded from this video."
        )

    # Repeat the last frame if the video is shorter than 16 frames.
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1].copy())

    sequence = np.asarray(
        frames[:SEQUENCE_LENGTH],
        dtype=np.float32,
    ) / 255.0

    model_input = np.expand_dims(
        sequence,
        axis=0,
    )

    # CNN-LSTM inference.
    try:
        probabilities = classifier.predict(
            model_input,
            verbose=0,
        )

    except Exception as error:
        raise RuntimeError(
            "CNN-LSTM inference failed. "
            "The saved model's Lambda attention layer may need "
            "a correction in cnn_lstm_model.py and retraining. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 2:
        probabilities = probabilities[0]

    if (
        probabilities.ndim != 1
        or len(probabilities) != len(CLASS_NAMES)
    ):
        raise RuntimeError(
            "Expected three class scores, but model output "
            f"shape was {probabilities.shape}."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    results = [
        {
            "Video": Path(video_path).name,
            "Predicted vigilance": predicted_class,
            "Model confidence (%)": round(
                confidence * 100,
                2,
            ),
            "Frames in sequence": SEQUENCE_LENGTH,
        }
    ]

    counts = {
        name: int(name == predicted_class)
        for name in CLASS_NAMES
    }

    return (
        results,
        counts,
        len(frames),
        SEQUENCE_LENGTH,
    )


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
    st.caption("Alarm: maximum 6 seconds")

    if classifier is not None:
        st.success("CNN-LSTM model loaded")
    else:
        st.error("CNN-LSTM model unavailable")


# ==========================================================
# MAIN HEADER
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
        1
        for item in incidents
        if item["status"] == "SUSTAINED_EYE_CLOSURE"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recorded incidents",
        len(incidents),
    )

    col2.metric(
        "Prolonged-closure events",
        sustained_events,
    )

    col3.metric(
        "Agent modules",
        "3",
    )

    st.caption(
        "Prolonged eye closure is an observation, "
        "not proof of drowsiness."
    )

    st.divider()

    st.subheader("System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### Vision Engine")
            st.write(
                "Webcam → Facial landmarks → EAR / MAR"
            )
            st.success(
                "Available when camera monitoring is active."
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### Agent Workflow")
            st.write("Monitor → Decision → Alert")
            st.success(
                "Monitor, decision, and alert modules are connected."
            )

    st.divider()

    st.subheader("Dataset Classifier")

    if classifier is not None:
        st.success(
            "The trained CNN-LSTM model is loaded."
        )
    else:
        st.warning(
            "The CNN-LSTM model is unavailable."
        )

        if model_error:
            st.code(model_error)

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
        st.info(
            "No incidents recorded yet. Start Live Monitoring."
        )


# ==========================================================
# LIVE MONITORING
# ==========================================================

elif page == "Live Monitoring":

    st.subheader("Live Driver Monitoring")

    st.caption(
        "Live webcam input with agent results."
    )

    st.warning(
        "This prototype detects possible prolonged eye closure. "
        "It is not a validated drowsiness classifier or "
        "vehicle safety system."
    )

    camera_context = webrtc_streamer(
        key="aegis_driver_camera",
        video_processor_factory=DriverProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.divider()

    st.subheader("Live Agent Results")

    @st.fragment(run_every=2)
    def live_agent_panel():

        processor = getattr(
            camera_context,
            "video_processor",
            None,
        )

        if processor is None:
            st.info(
                "Start the camera to see live agent results."
            )
            return

        monitor = getattr(
            processor,
            "latest_event",
            None,
        )

        if not monitor:
            st.info(
                "Waiting for the first webcam observation..."
            )
            return

        decision = DecisionAgent().decide(monitor)

        status = monitor.get(
            "status",
            "UNABLE_TO_ASSESS",
        )

        action = decision.get(
            "action",
            "WAIT",
        )

        priority = decision.get(
            "priority",
            "LOW",
        )

        event_id = monitor.get("event_id")

        alert_agent = st.session_state.alert_agent
        alert_result = None

        if status != "SUSTAINED_EYE_CLOSURE":

            if alert_agent.is_active():
                alert_agent.stop()

        elif (
            event_id is not None
            and event_id
            != st.session_state.last_alerted_event_id
        ):

            st.session_state.last_alerted_event_id = event_id

            alert_result = alert_agent.execute(
                {"alert_required": True}
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Monitor status",
            status.replace("_", " ").title(),
        )

        col2.metric(
            "Decision action",
            action.replace("_", " "),
        )

        col3.metric(
            "Priority",
            priority,
        )

        ear = monitor.get("ear")

        duration = (
            monitor.get("closed_duration", 0.0)
            or 0.0
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Eye Aspect Ratio (EAR)",
            f"{ear:.3f}" if ear is not None else "N/A",
        )

        col2.metric(
            "Eye-closure duration",
            f"{duration:.1f} s",
        )

        st.write(
            "**Monitor Agent:**",
            monitor.get("message", ""),
        )

        st.write(
            "**Decision Agent:**",
            decision.get("reason", ""),
        )

        if status == "SUSTAINED_EYE_CLOSURE":

            if (
                event_id is not None
                and event_id
                != st.session_state.last_logged_event_id
            ):

                save_incident(
                    monitor,
                    decision,
                )

                st.session_state.last_logged_event_id = event_id

            st.error(
                "Possible prolonged eye closure detected. "
                "Further assessment is required."
            )

        else:
            st.info(
                "No sustained eye-closure event currently detected."
            )

        if alert_agent.is_active():

            st.error("🔊 WARNING ALARM ACTIVE")

            if st.button(
                "Stop alarm now",
                key="stop_aegis_alarm",
            ):
                alert_agent.stop()
                st.success("Alarm stopped.")

        else:
            st.caption("Alarm is on standby.")

        if alert_result is not None:

            if isinstance(alert_result, dict):
                st.caption(
                    alert_result.get("message", "")
                )
            else:
                st.caption(str(alert_result))

        st.caption(
            "The alarm is designed to stop when the eyes reopen "
            "or after its maximum duration of 6 seconds."
        )

    live_agent_panel()


# ==========================================================
# DATASET TESTING
# ==========================================================

elif page == "Dataset Testing":

    st.subheader("UTA-RLDD Dataset Testing")

    st.caption(
        "Upload a prerecorded video and run the trained CNN-LSTM model."
    )

    st.warning(
        "Predictions are experimental and must not be used "
        "as real-world driving safety decisions."
    )

    if model_error:
        st.error("CNN-LSTM model loading error:")
        st.code(model_error)

    accuracy = metadata.get("test_accuracy")

    if accuracy is not None:
        st.caption(
            "Saved held-out test accuracy: "
            f"{float(accuracy) * 100:.2f}%"
        )

    uploaded_video = st.file_uploader(
        "Upload a dataset video",
        type=["mp4", "mov", "avi", "mkv"],
        key="dataset_video_upload",
    )

    if uploaded_video is None:

        st.info("Upload a video to begin.")

    else:

        suffix = (
            Path(uploaded_video.name).suffix
            or ".mp4"
        )

        video_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getbuffer()
                )

                video_path = Path(temp_file.name)

            st.markdown("### Video Preview")

            st.video(
                uploaded_video.getvalue()
            )

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():

                st.error(
                    "OpenCV could not open this video."
                )

            else:

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                frame_count = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                cap.release()

                duration = (
                    frame_count / fps
                    if fps > 0
                    else 0
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total frames",
                    f"{frame_count:,}",
                )

                col2.metric(
                    "Frame rate",
                    f"{fps:.2f} FPS"
                    if fps > 0
                    else "Unknown",
                )

                col3.metric(
                    "Duration",
                    f"{duration:.1f} seconds",
                )

                col4.metric(
                    "Resolution",
                    f"{width} × {height}",
                )

                if frame_count > 0:

                    frame_index = st.slider(
                        "Inspect a frame",
                        0,
                        frame_count - 1,
                        0,
                        key="dataset_frame_index",
                    )

                    preview_cap = cv2.VideoCapture(
                        str(video_path)
                    )

                    preview_cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(frame_index),
                    )

                    success, frame = preview_cap.read()
                    preview_cap.release()

                    if success:
                        st.image(
                            cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB,
                            ),
                            caption=f"Frame {frame_index}",
                            use_container_width=True,
                        )

                st.divider()

                st.markdown("### CNN-LSTM Prediction")

                if classifier is None:

                    st.error(
                        "The model is not available. "
                        "Fix the model-loading error before "
                        "running inference."
                    )

                elif st.button(
                    "Analyse video with trained model",
                    type="primary",
                    key="analyse_dataset_video",
                ):

                    progress = st.progress(
                        0.0,
                        text="Preparing video analysis...",
                    )

                    try:

                        (
                            results,
                            counts,
                            sampled,
                            detected,
                        ) = analyse_video(
                            video_path,
                            progress,
                        )

                        st.session_state.dataset_results = results
                        st.session_state.dataset_counts = counts
                        st.session_state.dataset_sampled = sampled
                        st.session_state.dataset_detected = detected
                        st.session_state.dataset_filename = (
                            uploaded_video.name
                        )

                        progress.progress(
                            1.0,
                            text="Analysis complete.",
                        )

                        st.success(
                            "Video analysis completed."
                        )

                    except Exception as error:

                        st.error(
                            f"Video analysis failed: {error}"
                        )

                    finally:
                        progress.empty()

                if "dataset_results" in st.session_state:

                    results = st.session_state.dataset_results
                    counts = st.session_state.dataset_counts
                    sampled = st.session_state.dataset_sampled
                    detected = st.session_state.dataset_detected

                    st.markdown("#### Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Frames sampled",
                        sampled,
                    )

                    col2.metric(
                        "Frames in sequence",
                        detected,
                    )

                    col3.metric(
                        "Video predictions",
                        len(results),
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("#### Class Results")

                    for class_name in CLASS_NAMES:
                        st.write(
                            f"**{class_name}:** "
                            f"{counts[class_name]} prediction(s)"
                        )

                    csv_data = pd.DataFrame(
                        results
                    ).to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download prediction results (CSV)",
                        data=csv_data,
                        file_name="aegis_dataset_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as error:

            st.error(
                f"Could not process the uploaded video: {error}"
            )

        finally:

            if video_path is not None:
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    pass


# ==========================================================
# AGENT CONTROL CENTRE
# ==========================================================

elif page == "Agent Control Centre":

    st.subheader("Agent Control Centre")

    st.caption(
        "Implemented modules and current integration status."
    )

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
            "Integration status depends on your orchestrator code.",
        ),
        (
            "Alert Agent",
            "agents/alert_agent.py",
            "Plays a warning tone for up to 6 seconds.",
            "Connected to live monitoring",
        ),
        (
            "CNN-LSTM Video Model",
            "models/aegis_cnn_lstm_attention.keras",
            "MobileNetV2 CNN + LSTM + temporal attention.",
            "Loaded" if classifier is not None else "Unavailable",
        ),
        (
            "Incident Database",
            "aegis_incidents.db",
            "Stores prolonged-eye-closure events.",
            "Connected",
        ),
    ]

    for (
        name,
        file_name,
        description,
        status_text,
    ) in agent_details:

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

        st.metric(
            "Total recorded incidents",
            len(incidents),
        )

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

        st.error(
            "The CNN-LSTM model could not be loaded."
        )

        if model_error:
            st.code(model_error)

        st.info(
            "Check models/aegis_cnn_lstm_attention.keras "
            "and your training script."
        )

    else:

        st.success(
            "CNN-LSTM video model loaded successfully."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Architecture",
            "CNN + LSTM",
        )

        col2.metric(
            "Input sequence",
            "16 frames",
        )

        col3.metric(
            "Classes",
            "3",
        )

        accuracy = metadata.get("test_accuracy")

        if accuracy is not None:
            st.metric(
                "Held-out test accuracy",
                f"{float(accuracy) * 100:.2f}%",
            )

        st.markdown("### Training Details")

        st.write(
            "**Architecture:** MobileNetV2 CNN + LSTM + temporal attention."
        )

        st.write(
            "**Input:** 16 RGB frames resized to 96 × 96 pixels."
        )

        st.write(
            "**Classes:** Alert, Low Vigilant, Drowsy."
        )

        for key, label in [
            ("usable_videos", "Usable videos"),
            ("training_videos", "Training videos"),
            ("validation_videos", "Validation videos"),
            ("test_videos", "Test videos"),
        ]:

            if metadata.get(key) is not None:
                st.write(
                    f"**{label}:** {metadata[key]}"
                )

        st.markdown("### Evaluation Interpretation")

        st.warning(
            "This model is experimental. Do not use it as "
            "the sole basis for real-world driving decisions."
        )

        st.write(
            "The CNN extracts visual features from frames, "
            "the LSTM models changes across the sequence, "
            "and temporal attention learns which frames to "
            "emphasize. Further validation is needed."
        )

        st.markdown("### Class Labels")

        st.dataframe(
            [
                {
                    "Label code": "0",
                    "Class": "Alert",
                },
                {
                    "Label code": "5",
                    "Class": "Low Vigilant",
                },
                {
                    "Label code": "10",
                    "Class": "Drowsy",
                },
            ],
            use_container_width=True,
            hide_index=True,
        )