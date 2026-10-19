
import cv2
import numpy as np
import mediapipe as mp
import joblib
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path("dataset_sample")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "0": "Alert",
    "5": "Low Vigilant",
    "10": "Drowsy"
}

CLASS_ORDER = ["0", "5", "10"]
SAMPLES_PER_VIDEO = 80
RANDOM_STATE = 42

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def extract_features(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    height, width = frame.shape[:2]

    def point(index):
        return (
            landmarks[index].x * width,
            landmarks[index].y * height
        )

    def calculate_ear(indices):
        p = [point(i) for i in indices]

        horizontal = distance(p[0], p[3])
        if horizontal < 1e-6:
            return np.nan

        return (
            distance(p[1], p[5]) +
            distance(p[2], p[4])
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

    features = [
        ear_value,
        mar_value,
        face_width,
        face_height
    ]

    if not np.all(np.isfinite(features)):
        return None

    return features


def video_features(video_path, face_mesh):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print(f"WARNING: No frames found in {video_path}")
        return []

    # Sample frames evenly across the clip.
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(SAMPLES_PER_VIDEO, total_frames),
        dtype=int
    )

    features = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        values = extract_features(frame, face_mesh)

        if values is not None:
            features.append(values)

    cap.release()
    return features


# ============================================================
# FIND ALL LABELLED VIDEOS
# ============================================================

if not DATASET.exists():
    raise SystemExit(
        f"Dataset folder not found: {DATASET.resolve()}"
    )

videos = sorted(
    [
        path for path in DATASET.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mov"}
        and path.stem in LABELS
    ],
    key=lambda path: str(path).lower()
)

if not videos:
    raise SystemExit(
        f"No labelled videos found in {DATASET.resolve()}"
    )

print(f"Found {len(videos)} labelled videos.")


# ============================================================
# EXTRACT DATA
# ============================================================

X, y, groups = [], [], []
usable_video_count = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    for video in videos:
        label_code = video.stem
        subject = video.parent.name

        print(
            f"\nSubject {subject} | "
            f"Class {LABELS[label_code]} | "
            f"Video {video.name}"
        )

        features = video_features(video, face_mesh)

        if not features:
            print("  No usable face frames; skipping video.")
            continue

        X.extend(features)
        y.extend([label_code] * len(features))
        groups.extend([subject] * len(features))

        usable_video_count += 1
        print(f"  Extracted {len(features)} face frames.")


if not X:
    raise SystemExit(
        "No usable facial features were extracted. "
        "Check the video paths and face visibility."
    )

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=str)
groups = np.asarray(groups, dtype=str)

subjects = sorted(np.unique(groups).tolist())

print("\n" + "=" * 55)
print("DATASET SUMMARY")
print("=" * 55)
print("Usable videos:", usable_video_count)
print("Total sampled face frames:", len(y))
print("Subjects:", subjects)

for code in CLASS_ORDER:
    print(
        f"{LABELS[code]}: "
        f"{int(np.sum(y == code))} frames"
    )

# Verify that all classes exist in the dataset.
if set(np.unique(y)) != set(CLASS_ORDER):
    raise SystemExit(
        "The extracted data does not contain all three classes. "
        "Check the dataset folders and filenames."
    )

# Each subject should have examples from all three classes
# for a fair leave-one-subject-out evaluation.
for subject in subjects:
    subject_classes = set(y[groups == subject])

    if subject_classes != set(CLASS_ORDER):
        print(
            f"WARNING: Subject {subject} does not contain "
            "all three classes."
        )


# ============================================================
# SUBJECT-WISE EVALUATION
# ============================================================

print("\n" + "=" * 55)
print("LEAVE-ONE-SUBJECT-OUT EVALUATION")
print("=" * 55)

cross_validation = LeaveOneGroupOut()
scores = []
oof_true = []
oof_pred = []

for train_indices, test_indices in cross_validation.split(
    X, y, groups
):
    train_classes = set(y[train_indices])
    test_subject = groups[test_indices][0]

    if train_classes != set(CLASS_ORDER):
        print(
            f"Skipping subject {test_subject}: "
            "training split does not contain all classes."
        )
        continue

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            class_weight=None,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1
        )
    )

    model.fit(X[train_indices], y[train_indices])
    predictions = model.predict(X[test_indices])

    accuracy = accuracy_score(
        y[test_indices],
        predictions
    )

    scores.append(accuracy)
    oof_true.extend(y[test_indices])
    oof_pred.extend(predictions)

    print(
        f"Test subject {test_subject}: "
        f"accuracy = {accuracy:.4f}, "
        f"frames = {len(test_indices)}"
    )

if not scores:
    raise SystemExit(
        "No valid evaluation folds were produced."
    )

mean_accuracy = float(np.mean(scores))

print("\nMean subject-wise accuracy:", round(mean_accuracy, 4))
print("\nClassification report:")

print(
    classification_report(
        oof_true,
        oof_pred,
        labels=CLASS_ORDER,
        target_names=[LABELS[c] for c in CLASS_ORDER],
        zero_division=0
    )
)


# ============================================================
# TRAIN FINAL DEMONSTRATION MODEL
# ============================================================

print("\nTraining final demonstration model on all extracted data...")

final_model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=150,
        class_weight=None,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1
    )
)

final_model.fit(X, y)

model_path = MODEL_DIR / "aegis_vigilance_model_v2.joblib"
metadata_path = MODEL_DIR / "aegis_vigilance_metadata_v2.json"

joblib.dump(final_model, model_path)

metadata = {
    "model_name": "AEGIS Vigilance Baseline",
    "algorithm": "Random Forest",
    "labels": LABELS,
    "features": [
        "EAR",
        "MAR",
        "face_width_ratio",
        "face_height_ratio"
    ],
    "training_subjects": subjects,
    "usable_video_count": usable_video_count,
    "frame_count": int(len(y)),
    "subject_wise_mean_accuracy": mean_accuracy,
    "evaluation": "Leave-one-subject-out cross-validation",
    "note": (
        "Preliminary research baseline. "
        "Not validated for real-world driver safety. "
        "Predictions are based on sampled facial frames."
    )
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)
print("Saved model:", model_path.resolve())
print("Saved metadata:", metadata_path.resolve())
print("Evaluation accuracy:", round(mean_accuracy, 4))

