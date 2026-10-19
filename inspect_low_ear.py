
import cv2
from pathlib import Path

VIDEO_PATH = r"C:\Users\Jigesha\UTA_RLDD_sample\sample_video.mp4"
OUTPUT_DIR = Path("low_ear_inspection")

timestamps = [104.64, 104.67, 104.70, 105.54, 111.28,
              113.81, 123.22, 133.53, 135.80]

OUTPUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)

for seconds in timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()

    if ok:
        filename = OUTPUT_DIR / f"frame_{seconds:.2f}s.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Saved {filename}")
    else:
        print(f"Could not read frame at {seconds:.2f}s")

cap.release()
print("Inspection frames saved.")
