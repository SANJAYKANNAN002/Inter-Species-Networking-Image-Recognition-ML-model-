from pathlib import Path

from ultralytics import YOLO


# Project root = parent of the app directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Use the corrected model with the official class-name mapping
MODEL_PATH = BASE_DIR / "model" / "best_fixed.pt"


class WildlifeDetector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

    def predict(self, source, confidence=0.25):
        """
        Run object detection on an image/video/frame.

        Returns the Ultralytics Results object.
        """
        results = self.model.predict(
            source=source,
            conf=confidence,
            device="cpu",
            verbose=False
        )

        return results