import pytest
import numpy as np
from src.detector import HandDetector

def test_detector_initialization():
    detector = HandDetector()
    assert detector is not None

def test_detect_no_hand():
    # Create a black image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = HandDetector(static_image_mode=True, min_detection_confidence=0.9)
    results, landmarks, bboxes, handedness = detector.detect(image)
    assert len(landmarks) == 0
    assert len(bboxes) == 0
    assert len(handedness) == 0
