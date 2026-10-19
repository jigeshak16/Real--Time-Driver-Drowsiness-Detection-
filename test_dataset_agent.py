
import cv2
import mediapipe as mp
import numpy as np

from agents.monitor_agent import MonitorAgent

# UTA-RLDD sample video
VIDEO_PATH = r"C:\Users\Jigesha\UTA_RLDD_sample\sample_video.mp4"

# Experimental settings — not clinically validated
EAR_THRESHOLD = 0.12
CLOSED_DURATION_THRESHOLD = 1.5

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def calculate_ear(landmarks, indices, width, height):
    points = [
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in indices
    ]

    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return None

    vertical = (
        distance(points[1], points[5])
        + distance(points[2], points[4])
    )

    return vertical / (2 * horizontal)


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS.")

print("Video opened successfully")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print("Processing EVERY frame...\n")

mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

agent = MonitorAgent(
    ear_threshold=EAR_THRESHOLD,
    duration_threshold=CLOSED_DURATION_THRESHOLD,
)

# Tracking variables
eyes_closed_since = None
frame_number = 0
faces_found = 0
frames_without_face = 0
events_found = 0
ear_values = []
below_threshold_count = 0
longest_closed_duration = 0.0
low_ear_samples = []

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        current_frame = frame_number
        frame_number += 1

        height, width = frame.shape[:2]
        timestamp = current_frame / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        # Diagnostic experiment: preserve the timer across missed faces.
        # Missing-face gaps are tracked separately because this can
        # overestimate closure duration.
        if not result.multi_face_landmarks:
            frames_without_face += 1
            continue

        faces_found += 1
        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(
            landmarks, LEFT_EYE, width, height
        )
        right_ear = calculate_ear(
            landmarks, RIGHT_EYE, width, height
        )

        if left_ear is None or right_ear is None:
            continue

        ear = float((left_ear + right_ear) / 2)
        ear_values.append(ear)

        if ear < EAR_THRESHOLD:
            below_threshold_count += 1
            low_ear_samples.append((timestamp, ear))

            if eyes_closed_since is None:
                eyes_closed_since = timestamp

            closed_duration = timestamp - eyes_closed_since
        else:
            eyes_closed_since = None
            closed_duration = 0.0

        longest_closed_duration = max(
            longest_closed_duration,
            closed_duration,
        )

        # Pass the measured EAR and estimated duration to the agent.
        observation = agent.analyse(
            ear=ear,
            closed_duration=float(closed_duration),
            face_detected=True,
        )

        if observation.get("possible_event", False):
            events_found += 1

        # Progress output every 10 seconds of video
        if current_frame % max(1, int(fps * 10)) == 0:
            print(
                f"Time={timestamp:6.1f}s | "
                f"EAR={ear:.3f} | "
                f"Closure={closed_duration:.2f}s | "
                f"Status={observation.get('status')}"
            )

finally:
    cap.release()
    mesh.close()


# Final summary — printed once
print("\n--- DATASET TEST SUMMARY ---")
print(f"Frames read: {frame_number}")
print(f"Frames with detected faces: {faces_found}")
print(f"Frames without detected faces: {frames_without_face}")
print(f"Possible-event observations: {events_found}")

if ear_values:
    print(f"EAR samples: {len(ear_values)}")
    print(f"Lowest EAR: {min(ear_values):.3f}")
    print(f"Highest EAR: {max(ear_values):.3f}")
    print(f"Average EAR: {np.mean(ear_values):.3f}")
else:
    print("No EAR measurements were collected.")

print(f"EAR below threshold: {below_threshold_count}")
print(
    f"Longest tracked closure: "
    f"{longest_closed_duration:.2f} seconds"
)

print("\n--- LOW EAR SAMPLES (first 20) ---")
if low_ear_samples:
    for timestamp, ear in low_ear_samples[:20]:
        print(f"Time={timestamp:.2f}s | EAR={ear:.3f}")
else:
    print("No EAR samples fell below the threshold.")

print("\nTest complete.")
