
import subprocess
import imageio_ffmpeg
import tempfile
import os
from pathlib import Path
from zipfile import ZipFile
import json
import re

import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from cnn_lstm_model import build_model


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_sample"
MODEL_DIR = BASE_DIR / "models"

ZIP_FILES = [
    Path(r"C:\Users\Jigesha\Downloads\Fold3_part1.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold3_part2.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold4_part1.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold4_part2.zip"),
]

SEQUENCE_LENGTH = 16
NUM_SEQUENCES_PER_VIDEO = 1
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

EPOCHS = 8
BATCH_SIZE = 8
RANDOM_SEED = 42

LABEL_MAP = {"0": 0, "5": 1, "10": 2}
CLASS_NAMES = ["Alert", "Low Vigilant", "Drowsy"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==================== LABELS AND SUBJECTS ====================

def get_label(stem):
    match = re.fullmatch(r"(0|5|10)(?:_\d+)?", stem)
    return LABEL_MAP[match.group(1)] if match else None


def get_fold_name(folder_name):
    return re.sub(r"_part\d+$", "", folder_name, flags=re.IGNORECASE)


def make_group_id(fold_folder, subject_id):
    return f"{get_fold_name(fold_folder)}_{subject_id}"


# ==================== FIND VIDEOS ====================

def find_videos():
    videos = []

    if DATASET_DIR.exists():
        for path in DATASET_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            label = get_label(path.stem)
            if label is None:
                continue

            relative = path.relative_to(DATASET_DIR)
            parts = relative.parts

            subject_id = path.parent.name
            fold_folder = parts[0] if len(parts) >= 3 else "dataset_sample"

            videos.append({
                "source": "file",
                "path": path,
                "archive": None,
                "member": None,
                "label": label,
                "group": make_group_id(fold_folder, subject_id),
                "name": str(path),
            })
    else:
        print(f"Warning: dataset folder not found: {DATASET_DIR}")

    for zip_path in ZIP_FILES:
        if not zip_path.exists():
            print(f"Warning: ZIP not found: {zip_path}")
            continue

        with ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                p = Path(member)

                if p.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                label = get_label(p.stem)
                if label is None:
                    continue

                parts = p.parts
                if len(parts) < 3:
                    print(f"Skipping unexpected ZIP path: {member}")
                    continue

                videos.append({
                    "source": "zip",
                    "path": None,
                    "archive": zip_path,
                    "member": member,
                    "label": label,
                    "group": make_group_id(parts[0], parts[-2]),
                    "name": f"{zip_path.name}:{member}",
                })

    unique = {}
    for video in videos:
        if video["source"] == "file":
            key = str(video["path"]).lower()
        else:
            key = f'{video["archive"]}|{video["member"]}'.lower()
        unique[key] = video

    return sorted(unique.values(), key=lambda v: v["name"].lower())


# ==================== VIDEO DECODING ====================

def resize_rgb_frame(frame):
    return cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))


def extract_sequences(video_path):
    """Decode local videos with OpenCV."""
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)

    try:
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            return None

        indices = np.linspace(
            0, frame_count - 1, SEQUENCE_LENGTH
        ).astype(int)

        frames = []

        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(resize_rgb_frame(frame))

        return np.asarray(frames, dtype=np.uint8)[None, ...]

    finally:
        cap.release()


def extract_sequences_from_zip(archive, member):
    """
    Extract one ZIP video to a temporary file and decode it using FFmpeg.
    Avoids passing huge compressed video files through stdin.
    """
    temp_path = None

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        suffix = Path(member).suffix.lower() or ".mp4"

        # Extract in chunks so the entire video is not held in RAM.
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp_path = temp.name

            with archive.open(member, "r") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)

        video_filter = (
            f"fps=2,"
            f"scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={IMAGE_WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        )

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", temp_path,
            "-vf", video_filter,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
        )

        frame_size = IMAGE_HEIGHT * IMAGE_WIDTH * 3
        frame_count = len(result.stdout) // frame_size

        if frame_count == 0:
            error = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                error[:500] or "FFmpeg produced no frames."
            )

        # A nonzero exit code means decoding may have been incomplete.
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FFmpeg exited with code {result.returncode}: "
                f"{error[:300]}"
            )

        usable_bytes = frame_count * frame_size

        frames = np.frombuffer(
            result.stdout[:usable_bytes], dtype=np.uint8
        ).reshape(
            frame_count, IMAGE_HEIGHT, IMAGE_WIDTH, 3
        )

        indices = np.linspace(
            0, frame_count - 1, SEQUENCE_LENGTH
        ).astype(int)

        return frames[indices][None, ...].copy()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def load_video_sequences(video):
    if video["source"] == "file":
        return extract_sequences(video["path"])

    with ZipFile(video["archive"], "r") as archive:
        return extract_sequences_from_zip(
            archive, video["member"]
        )


# ==================== DATA SPLITTING ====================

def print_class_counts(title, labels):
    counts = np.bincount(labels, minlength=3)
    print(title)

    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[i]}")


def split_subjects(y, groups, test_size, seed):
    dummy_x = np.zeros((len(y), 1), dtype=np.float32)

    for attempt in range(300):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed + attempt,
        )

        left_idx, right_idx = next(
            splitter.split(dummy_x, y, groups=groups)
        )

        if (
            len(np.unique(y[left_idx])) == 3
            and len(np.unique(y[right_idx])) == 3
        ):
            return left_idx, right_idx

    raise RuntimeError(
        "Could not create subject-wise splits containing all classes. "
        "Check the labels and class coverage across subjects."
    )


def make_dataset(sequences, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (sequences, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(labels),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==================== MAIN TRAINING ====================

def main():
    videos = find_videos()

    if not videos:
        raise RuntimeError("No labeled videos found.")

    print(f"Found {len(videos)} labeled video entries.")

    print_class_counts(
        "Expected class counts:",
        np.asarray([v["label"] for v in videos], dtype=np.int32),
    )

    X_list, y_list, group_list, name_list = [], [], [], []

    for index, video in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {video['name']}")

        try:
            sequences = load_video_sequences(video)
        except Exception as error:
            print(f"  Skipped: {error}")
            continue

        if sequences is None:
            print("  Skipped: unable to decode video.")
            continue

        expected_shape = (
            NUM_SEQUENCES_PER_VIDEO,
            SEQUENCE_LENGTH,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        )

        if sequences.shape != expected_shape:
            print(f"  Skipped: unexpected frame shape {sequences.shape}")
            continue

        X_list.append(sequences)
        y_list.append(video["label"])
        group_list.append(video["group"])
        name_list.append(video["name"])

        print("  Decoded successfully.")

    if len(X_list) < 12:
        raise RuntimeError(
            f"Only {len(X_list)} usable videos found. "
            "Check decoder errors before training."
        )

    X = np.asarray(X_list, dtype=np.uint8)
    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(group_list)

    print("\nSequence array shape:", X.shape)
    print_class_counts("Usable class counts:", y)
    print("Unique subject groups:", len(np.unique(groups)))

    if len(np.unique(y)) != 3:
        raise RuntimeError("Not all three classes are represented.")

    trainval_idx, test_idx = split_subjects(
        y, groups, test_size=0.25, seed=RANDOM_SEED
    )

    fit_local, val_local = split_subjects(
        y[trainval_idx],
        groups[trainval_idx],
        test_size=0.25,
        seed=RANDOM_SEED + 100,
    )

    fit_idx = trainval_idx[fit_local]
    val_idx = trainval_idx[val_local]

    fit_groups = set(groups[fit_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert fit_groups.isdisjoint(val_groups)
    assert fit_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print("\nSplit summary:")
    print(f"Training videos: {len(fit_idx)}")
    print(f"Validation videos: {len(val_idx)}")
    print(f"Test videos: {len(test_idx)}")
    print(f"Training subjects: {len(fit_groups)}")
    print(f"Validation subjects: {len(val_groups)}")
    print(f"Test subjects: {len(test_groups)}")

    print_class_counts("Training classes:", y[fit_idx])
    print_class_counts("Validation classes:", y[val_idx])
    print_class_counts("Test classes:", y[test_idx])

    X_fit = X[fit_idx].reshape(
        -1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3
    )
    y_fit = y[fit_idx]

    X_val = X[val_idx].reshape(
        -1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3
    )
    y_val = y[val_idx]

    train_ds = make_dataset(X_fit, y_fit, training=True)
    val_ds = make_dataset(X_val, y_val)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        num_classes=3,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "aegis_best_checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    X_test = X[test_idx].reshape(
        -1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3
    )
    test_ds = make_dataset(
        X_test, np.zeros(len(X_test), dtype=np.int32)
    )

    probabilities = model.predict(test_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    y_test = y[test_idx]

    print("\nFinal held-out TEST classification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    test_accuracy = float(np.mean(predictions == y_test))

    model_path = MODEL_DIR / "aegis_cnn_lstm_attention.keras"
    model.save(model_path)

    metadata = {
        "architecture": "MobileNetV2-CNN-LSTM-Temporal-Attention",
        "sequence_length": SEQUENCE_LENGTH,
        "sequences_per_video": NUM_SEQUENCES_PER_VIDEO,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "class_names": CLASS_NAMES,
        "label_mapping": {
            "0": "Alert",
            "5": "Low Vigilant",
            "10": "Drowsy",
        },
        "usable_videos": len(X),
        "training_videos": len(fit_idx),
        "validation_videos": len(val_idx),
        "test_videos": len(test_idx),
        "training_subject_groups": sorted(fit_groups),
        "validation_subject_groups": sorted(val_groups),
        "test_subject_groups": sorted(test_groups),
        "test_accuracy": test_accuracy,
        "test_video_names": [name_list[i] for i in test_idx],
        "note": (
            "Preliminary subject-wise holdout results on a small dataset. "
            "Not a clinical or deployment reliability claim."
        ),
    }

    metadata_path = MODEL_DIR / "aegis_cnn_lstm_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()




