
from agents.monitor_agent import MonitorAgent
from agents.decision_agent import DecisionAgent


class AgentOrchestrator:
    """
    Coordinates the monitoring and decision agents.
    """

    def __init__(self):
        self.monitor_agent = MonitorAgent()
        self.decision_agent = DecisionAgent()

    def process(self, ear, closed_duration, face_detected=True):
        # Agent 1: Analyse webcam measurements
        monitor_result = self.monitor_agent.analyse(
            ear=ear,
            closed_duration=closed_duration,
            face_detected=face_detected,
        )

        # Agent 2: Decide what action should be taken
        decision_result = self.decision_agent.decide(
            monitor_result
        )

        # Combined result for dashboard and incident logging
        return {
            "monitor": monitor_result,
            "decision": decision_result,
            "timestamp": monitor_result.get("timestamp"),
        }
