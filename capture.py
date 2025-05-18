import cv2
import numpy as np
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamCapture:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self.is_initialized = False

    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open webcam with device ID {self.device_id}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to start webcam: {str(e)}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_initialized:
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def stop(self) -> bool:
        if not self.is_initialized:
            return True

        try:
            self.cap.release()
            self.is_initialized = False
            return True
        except Exception as e:
            logger.error(f"Error stopping webcam: {str(e)}")
            self.is_initialized = False
            return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()