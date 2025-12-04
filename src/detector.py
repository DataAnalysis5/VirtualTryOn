import mediapipe as mp
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, image: np.ndarray):
        """
        Detects hands in the image.
        Args:
            image: BGR image.
        Returns:
            results: MediaPipe results object.
            landmarks_list: List of landmarks (normalized x, y, z) for each detected hand.
            bboxes: List of bounding boxes [x_min, y_min, x_max, y_max] for each detected hand.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks_list = []
        bboxes = []

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                landmarks_list.append(landmarks)
                # Add some padding to bbox
                pad = 20
                bboxes.append([max(0, x_min - pad), max(0, y_min - pad), min(w, x_max + pad), min(h, y_max + pad)])
        
        handedness_list = []
        if results.multi_handedness:
            for h_info in results.multi_handedness:
                handedness_list.append(h_info.classification[0].label)
        
        return results, landmarks_list, bboxes, handedness_list

    @staticmethod
    def is_palm_facing(landmarks, handedness):
        """
        Determines if the palm is facing the camera.
        Args:
            landmarks: List of (x, y, z) normalized.
            handedness: 'Left' or 'Right'.
        Returns:
            bool: True if palm is facing camera.
        """
        # Wrist: 0, Index MCP: 5, Pinky MCP: 17
        p0 = np.array(landmarks[0][:2])
        p5 = np.array(landmarks[5][:2])
        p17 = np.array(landmarks[17][:2])
        
        # Vectors
        v1 = p5 - p0
        v2 = p17 - p0
        
        # Cross product z-component (x1*y2 - y1*x2)
        # Note: Image coordinates y is down.
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Heuristic based on winding order
        # Left Hand Label (e.g. mirrored Right hand): Palm if Cross > 0
        # Right Hand Label: Palm if Cross < 0
        if handedness == 'Right':
            return cross < 0
        else:
            return cross > 0

    def draw_landmarks(self, image: np.ndarray, results):
        """Draws landmarks on the image."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image
