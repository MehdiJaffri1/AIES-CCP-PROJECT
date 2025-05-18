import time
import cv2
from detection_system.models import DetectionModel
from detection_system.processor import FrameProcessor
from detection_system.capture import WebcamCapture

class DetectionApp:
    def __init__(
        self,
        currency_model_path: str,
        object_model_path: str,
        conf_threshold: float = 0.25
    ):
        self.currency_model = DetectionModel(currency_model_path)
        self.object_model = DetectionModel(object_model_path)
        self.processor = FrameProcessor(
            self.currency_model,
            self.object_model,
            conf_threshold
        )

    def run_detection(self, fps_target: int = 10) -> None:
        capture = WebcamCapture()
        print("Initializing webcam...")

        try:
            capture.start()
            print("Webcam started successfully")

            while True:
                try:
                    frame = capture.read_frame()
                    if frame is None:
                        continue

                    processed_frame, results = self.processor.process_frame(frame)

                    cv2.imshow("Detection", processed_frame)
                    cv2.waitKey(1)

                    if results.currency or results.objects:
                        print("\nDetections:")
                        if results.currency:
                            print("Currency:")
                            for det in results.currency:
                                print(f"  - {det.class_name} (Confidence: {det.confidence:.2f})")
                        if results.objects:
                            print("Objects:")
                            for det in results.objects:
                                print(f"  - {det.class_name} (Confidence: {det.confidence:.2f})")

                    print(f"\nFPS: {fps}")
                    time.sleep(max(0, 1/fps_target - (time.time() - current_time)))

                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    time.sleep(0.1)
                    continue

        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            try:
                capture.stop()
                print("Webcam stopped successfully")
            except Exception as e:
                print(f"Error stopping webcam: {str(e)}")