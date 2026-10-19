
from datetime import datetime


class MonitorAgent:
    """
    Reads live eye measurements and identifies
    possible eye-closure events.
    """

    def __init__(self, ear_threshold=0.12, duration_threshold=1.5):
        self.ear_threshold = ear_threshold
        self.duration_threshold = duration_threshold

    def analyse(self, ear, closed_duration, face_detected=True):
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "face_detected": bool(face_detected),
            "ear": round(float(ear), 4) if ear is not None else None,
            "closed_duration": round(float(closed_duration), 2),
            "status": "NORMAL",
            "possible_event": False,
            "message": "No sustained eye closure detected."
        }

        if not face_detected:
            event["status"] = "FACE_NOT_DETECTED"
            event["message"] = "Face is not visible. Check camera position."

        elif ear is None:
            event["status"] = "UNABLE_TO_ASSESS"
            event["message"] = "Eye measurements are unavailable."

        elif ear < self.ear_threshold:
            if closed_duration >= self.duration_threshold:
                event["status"] = "SUSTAINED_EYE_CLOSURE"
                event["possible_event"] = True
                event["message"] = (
                    "Possible prolonged eye closure. "
                    "Further assessment is required."
                )
            else:
                event["status"] = "EYES_APPEAR_CLOSED"
                event["message"] = "Eyes appear closed; monitoring continues."

        return event
