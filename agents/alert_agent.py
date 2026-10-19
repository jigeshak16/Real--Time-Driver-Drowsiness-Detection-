
import threading
import time

import numpy as np
import sounddevice as sd


class AlertAgent:
    def __init__(self, cooldown_seconds=8, alarm_duration=6):
        self.cooldown_seconds = cooldown_seconds
        self.alarm_duration = alarm_duration

        self.last_alert_time = 0
        self.lock = threading.Lock()
        self.alarm_active = False
        self.stop_event = threading.Event()
        self.audio_thread = None

        self.sample_rate = 44100
        self.alarm_sound = self._create_alarm()

    def _create_alarm(self):
        # Sharp alternating emergency warning tone
        sample_rate = self.sample_rate
        duration = 0.32

        t = np.arange(
            int(sample_rate * duration),
            dtype=np.float32,
        ) / sample_rate

        frequency = np.where(
            (t % 0.16) < 0.08,
            2400.0,
            1700.0,
        )

        phase = (
            2 * np.pi * np.cumsum(frequency) / sample_rate
        )
        sound = 0.65 * np.sin(phase)

        # Fade the beginning and end to reduce clicking
        fade = int(0.004 * sample_rate)
        sound[:fade] *= np.linspace(0, 1, fade)
        sound[-fade:] *= np.linspace(1, 0, fade)

        return sound.astype(np.float32)

    def _ring_for_six_seconds(self):
        deadline = time.monotonic() + self.alarm_duration

        try:
            while (
                not self.stop_event.is_set()
                and time.monotonic() < deadline
            ):
                sd.play(
                    self.alarm_sound,
                    samplerate=self.sample_rate,
                    blocking=True,
                )

        except Exception as error:
            print("Alarm playback error:", error)

        finally:
            sd.stop()

            with self.lock:
                self.alarm_active = False

    def execute(self, decision):
        if not decision.get("alert_required", False):
            return {
                "played": False,
                "message": "No alarm required.",
            }

        with self.lock:
            if self.alarm_active:
                return {
                    "played": False,
                    "message": "Alarm is already ringing.",
                }

            if (
                time.monotonic() - self.last_alert_time
                < self.cooldown_seconds
            ):
                return {
                    "played": False,
                    "message": "Alarm cooldown is active.",
                }

            self.last_alert_time = time.monotonic()
            self.stop_event.clear()
            self.alarm_active = True

            self.audio_thread = threading.Thread(
                target=self._ring_for_six_seconds,
                daemon=True,
            )
            self.audio_thread.start()

        return {
            "played": True,
            "message": "Warning alarm started for 6 seconds.",
        }

    def stop(self):
        self.stop_event.set()
        sd.stop()

        return {
            "stopped": True,
            "message": "Alarm stopped manually.",
        }

    def is_active(self):
        with self.lock:
            return self.alarm_active
