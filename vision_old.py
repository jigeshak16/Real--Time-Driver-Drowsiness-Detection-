
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase


class DriverProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils

    @staticmethod
    def distance(points, a, b, width, height):
        p1 = np.array([
            points[a].x * width,
            points[a].y * height,
        ])
        p2 = np.array([
            points[b].x * width,
            points[b].y * height,
        ])
        return np.linalg.norm(p1 - p2)

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
                connection_drawing_spec=(
                    self.drawer.DrawingSpec(
                        color=(0, 220, 180),
                        thickness=1,
                        circle_radius=1,
                    )
                ),
            )

            # Left eye aspect ratio
            left_vertical = (
                self.distance(points, 160, 144, width, height)
                + self.distance(points, 158, 153, width, height)
            )
            left_horizontal = self.distance(
                points, 33, 133, width, height
            )

            # Right eye aspect ratio
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

            cv2.putText(
                image,
                f"Face detected | EAR: {ear:.3f}"
                if ear is not None
                else "Face detected | EAR: unavailable",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 180),
                2,
            )

            cv2.putText(
                image,
                f"MAR: {mar:.3f}"
                if mar is not None
                else "MAR: unavailable",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 180),
                2,
            )

        else:
            cv2.putText(
                image,
                "Unable to assess: face not detected",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 100, 255),
                2,
            )

        return frame.from_ndarray(image, format="bgr24")
