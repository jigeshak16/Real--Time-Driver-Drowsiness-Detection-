
class DecisionAgent:
    """Selects an action based on monitoring observations."""

    def decide(self, monitor_result):
        status = monitor_result.get("status", "UNABLE_TO_ASSESS")
        duration = float(monitor_result.get("closed_duration") or 0)
        possible_event = monitor_result.get("possible_event", False)

        if status == "FACE_NOT_DETECTED":
            return {
                "action": "REQUEST_FACE_POSITION",
                "priority": "MEDIUM",
                "reason": "Face not visible. Adjust the camera.",
                "alert_required": False,
            }

        if status == "UNABLE_TO_ASSESS":
            return {
                "action": "RETRY_MONITORING",
                "priority": "LOW",
                "reason": "Eye measurements are unavailable.",
                "alert_required": False,
            }

        if possible_event and duration >= 5.0:
            return {
                "action": "ISSUE_ALERT",
                "priority": "HIGH",
                "reason": "Prolonged eye closure requires attention.",
                "alert_required": True,
            }

        if possible_event or status == "SUSTAINED_EYE_CLOSURE":
            return {
                "action": "ISSUE_ALERT",
                "priority": "MEDIUM",
                "reason": "Possible sustained eye closure. Continue assessment.",
                "alert_required": True,
            }

        if status == "EYES_APPEAR_CLOSED":
            return {
                "action": "CONTINUE_MONITORING",
                "priority": "LOW",
                "reason": "Brief eye closure observed.",
                "alert_required": False,
            }

        return {
            "action": "CONTINUE_MONITORING",
            "priority": "LOW",
            "reason": "No sustained eye closure detected.",
            "alert_required": False,
        }

