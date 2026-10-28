
from pathlib import Path
import cv2

DATASET_DIR = Path(__file__).resolve().parent / "dataset_sample"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
LABELS = {"0": "Alert", "5": "Low Vigilant", "10": "Drowsy"}

total = 0
failed = 0
class_counts = {name: 0 for name in LABELS.values()}

for path in sorted(DATASET_DIR.rglob("*")):
    if not path.is_file():
        continue

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        continue

    if path.stem not in LABELS:
        continue

    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
    duration = frames / fps if fps > 0 else 0
    cap.release()

    total += 1
    label = LABELS[path.stem]
    class_counts[label] += 1

    status = "OK" if opened and frames > 0 and fps > 0 else "CHECK"
    if status == "CHECK":
        failed += 1

    print(
        f"{status} | {label:14} | "
        f"frames={frames:6} | "
        f"duration={duration:7.2f}s | {path}"
    )

print("\n---------- DATASET SUMMARY ----------")
print("Total labeled videos:", total)
print("Videos needing inspection:", failed)

for label, count in class_counts.items():
    print(f"{label}: {count}")
