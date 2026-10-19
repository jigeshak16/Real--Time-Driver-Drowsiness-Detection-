
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from streamlit_webrtc import VideoProcessorBase

from agents.monitor_agent import MonitorAgent


class DriverProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.drawer = mp.solutions.drawing_utils
        self.ear_history = deque(maxlen=5)
        self.eyes_closed_since = None

        # Experimental prototype settings
        self.ear_threshold = 0.12
        self.closed_duration_threshold = 1.5

        # Monitor Agent
        self.monitor_agent = MonitorAgent(
            ear_threshold=self.ear_threshold,
            duration_threshold=self.closed_duration_threshold,
        )

        # Latest observation shared with the dashboard
        self.latest_event = {
            "status": "INITIALISING",
            "possible_event": False,
            "message": "Waiting for camera input.",
            "event_id": None,
        }

    @staticmethod
    def distance(points, a, b, width, height):
        p1 = np.array([points[a].x * width, points[a].y * height])
        p2 = np.array([points[b].x * width, points[b].y * height])
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def draw_panel(
        image,
        text,
        y,
        color=(245, 248, 255),
        scale=0.7,
        thickness=2,
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = 24
        height, width = image.shape[:2]

        max_text_width = max(100, width - x - 35)

        while scale > 0.4:
            text_width = cv2.getTextSize(
                text, font, scale, thickness
            )[0][0]
            if text_width <= max_text_width:
                break
            scale -= 0.05

        text_size = cv2.getTextSize(
            text, font, scale, thickness
        )[0]

        top = max(0, y - 29)
        bottom = min(height - 1, y + 11)
        right = min(width - 1, x + text_size[0] + 22)

        cv2.rectangle(
            image,
            (max(0, x - 12), top),
            (right, bottom),
            (18, 27, 43),
            -1,
        )

        cv2.putText(
            image,
            text,
            (x, min(height - 5, y)),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            points = face.landmark

            self.drawer.draw_landmarks(
                image=image,
                landmark_list=face,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawer.DrawingSpec(
                    color=(0, 220, 180),
                    thickness=1,
                    circle_radius=1,
                ),
            )

            # Left eye measurements
            left_vertical = (
                self.distance(points, 160, 144, width, height)
                + self.distance(points, 158, 153, width, height)
            )
            left_horizontal = self.distance(
                points, 33, 133, width, height
            )

            # Right eye measurements
            right_vertical = (
                self.distance(points, 385, 380, width, height)
                + self.distance(points, 387, 373, width, height)
            )
            right_horizontal = self.distance(
                points, 362, 263, width, height
            )

            ear = None

            if left_horizontal > 0 and right_horizontal > 0:
                ear = (
                    left_vertical / (2 * left_horizontal)
                    + right_vertical / (2 * right_horizontal)
                ) / 2

            # Mouth aspect ratio
            mouth_vertical = self.distance(
                points, 13, 14, width, height
            )
            mouth_horizontal = self.distance(
                points, 78, 308, width, height
            )

            mar = (
                mouth_vertical / mouth_horizontal
                if mouth_horizontal > 0
                else None
            )

            # Smooth EAR across recent frames
            if ear is not None:
                self.ear_history.append(ear)
                smoothed_ear = float(np.mean(self.ear_history))
                now = time.monotonic()

                if smoothed_ear < self.ear_threshold:
                    if self.eyes_closed_since is None:
                        self.eyes_closed_since = now

                    closed_duration = now - self.eyes_closed_since
                else:
                    self.eyes_closed_since = None
                    closed_duration = 0.0

            else:
                self.ear_history.clear()
                self.eyes_closed_since = None
                closed_duration = 0.0
                smoothed_ear = None

            # Run Monitor Agent on live webcam measurements
            self.latest_event = self.monitor_agent.analyse(
                ear=smoothed_ear,
                closed_duration=closed_duration,
                face_detected=True,
            )

            # Stable ID for one continuous sustained closure
            if (
                self.latest_event["status"] == "SUSTAINED_EYE_CLOSURE"
                and self.eyes_closed_since is not None
            ):
                self.latest_event["event_id"] = (
                    f"closure_{self.eyes_closed_since:.6f}"
                )
            else:
                self.latest_event["event_id"] = None

            status = self.latest_event["status"]

            if status == "NORMAL":
                state_color = (90, 235, 150)
                display_status = "EYES OPEN / NORMAL"
            elif status == "EYES_APPEAR_CLOSED":
                state_color = (0, 190, 255)
                display_status = "EYES APPEAR CLOSED"
            elif status == "SUSTAINED_EYE_CLOSURE":
                state_color = (70, 80, 255)
                display_status = "POSSIBLE PROLONGED CLOSURE"
            else:
                state_color = (0, 190, 255)
                display_status = status.replace("_", " ")

            # Live camera overlay
            self.draw_panel(
                image,
                "DRIVER MONITORING",
                40,
                color=(255, 255, 255),
                scale=0.8,
            )

            self.draw_panel(
                image,
                f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A",
                82,
                color=(90, 235, 150),
            )

            self.draw_panel(
                image,
                f"MAR: {mar:.3f}" if mar is not None else "MAR: N/A",
                122,
                color=(100, 210, 255),
            )

            self.draw_panel(
                image,
                f"AGENT: {display_status}",
                164,
                color=state_color,
                scale=0.65,
            )

            self.draw_panel(
                image,
                f"Eye closure: {closed_duration:.1f}s",
                204,
                color=(235, 240, 250),
            )

            if self.latest_event["possible_event"]:
                self.draw_panel(
                    image,
                    "CHECK DRIVER ALERTNESS",
                    245,
                    color=(80, 100, 255),
                    scale=0.6,
                )

        else:
            self.ear_history.clear()
            self.eyes_closed_since = None

            # Report missing face to Monitor Agent
            self.latest_event = self.monitor_agent.analyse(
                ear=None,
                closed_duration=0.0,
                face_detected=False,
            )

            # No active event when the face cannot be detected
            self.latest_event["event_id"] = None

            self.draw_panel(
                image,
                "FACE NOT DETECTED",
                45,
                color=(80, 100, 255),
                scale=0.8,
            )

            self.draw_panel(
                image,
                "Please face the camera",
                90,
                color=(245, 248, 255),
                scale=0.65,
            )

            self.draw_panel(
                image,
                "AGENT: WAITING FOR FACE",
                135,
                color=(0, 190, 255),
                scale=0.65,
            )

        return frame.from_ndarray(image, format="bgr24")




