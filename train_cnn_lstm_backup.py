
from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from cnn_lstm_model import build_model


# ---------- CONFIGURATION ----------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_sample"
MODEL_DIR = BASE_DIR / "models"

SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
EPOCHS = 15
BATCH_SIZE = 4

# Original video filename -> model class index
LABEL_MAP = {
    "0": 0,   # Alert
    "5": 1,   # Low Vigilant
    "10": 2   # Drowsy
}

CLASS_NAMES = ["Alert", "Low Vigilant", "Drowsy"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_sequence(video_path):
    """Extract 16 evenly spaced RGB frames from a video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 1:
        cap.release()
        return None

    frame_indices = np.linspace(
        0, frame_count - 1, SEQUENCE_LENGTH
    ).astype(int)

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.asarray(frames, dtype=np.float32)


def find_videos():
    """Find labeled videos and infer subject IDs from folder names."""
    videos = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    for path in DATASET_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        if path.stem not in LABEL_MAP:
            continue

        # Expected structure: dataset_sample/fold/subject/0.mp4
        subject_id = path.parent.name

        # Include fold in ID so subjects in different folds
        # are treated as separate groups if folder IDs repeat.
        fold_id = path.parent.parent.name
        group_id = f"{fold_id}_{subject_id}"

        videos.append((path, LABEL_MAP[path.stem], group_id))

    return videos


def main():
    videos = find_videos()

    if not videos:
        raise RuntimeError(
            "No labeled videos found. Expected filenames "
            "0.mp4, 5.mp4, and 10.mp4 (or .MOV) under dataset_sample."
        )

    print(f"Found {len(videos)} labeled videos.")

    X, y, groups = [], [], []

    for index, (video_path, label, group_id) in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] Reading {video_path}")

        sequence = extract_sequence(video_path)

        if sequence is None:
            print("  Skipped: could not extract frames.")
            continue

        X.append(sequence)
        y.append(label)
        groups.append(group_id)

    if len(X) < 6:
        raise RuntimeError(
            f"Only {len(X)} usable videos found. "
            "More labeled videos are needed for a meaningful split."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)

    print("Sequence array shape:", X.shape)
    print("Class counts:", {
        name: int(np.sum(y == i))
        for i, name in enumerate(CLASS_NAMES)
    })
    print("Unique subject groups:", len(np.unique(groups)))

    # Keep subjects out of both training and test sets.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )
    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if len(np.unique(y_train)) < 3:
        raise RuntimeError(
            "The training split does not contain all three classes. "
            "More subjects/videos or a different split are needed."
        )

    if len(np.unique(y_test)) < 3:
        print(
            "WARNING: Test split does not contain all three classes; "
            "metrics may be incomplete."
        )

    model = build_model(
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        num_classes=3
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "aegis_cnn_lstm_attention.keras"
    model.save(model_path)

    metadata = {
        "architecture": "CNN-LSTM-Attention",
        "sequence_length": SEQUENCE_LENGTH,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "class_names": CLASS_NAMES,
        "label_mapping": {"0": "Alert", "5": "Low Vigilant", "10": "Drowsy"},
        "usable_videos": len(X),
        "training_videos": len(train_indices),
        "test_videos": len(test_indices),
        "training_subject_groups": len(np.unique(groups[train_indices])),
        "test_subject_groups": len(np.unique(groups[test_indices])),
        "note": "Small-dataset holdout results are preliminary."
    }

    metadata_path = MODEL_DIR / "aegis_cnn_lstm_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
