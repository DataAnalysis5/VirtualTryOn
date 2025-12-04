import pytest
import numpy as np
from src.ring_fitter import compute_ring_transform

def test_compute_ring_transform_valid():
    # Create dummy landmarks
    # 21 landmarks, (x, y, z)
    landmarks = [(0.5, 0.5, 0.0)] * 21
    
    # Set specific landmarks for ring finger (indices 13, 14, 15, 16)
    # MCP (13) at (0.5, 0.5)
    # PIP (14) at (0.5, 0.4) -> Vector pointing up (0, -0.1)
    landmarks[13] = (0.5, 0.5, 0.0)
    landmarks[14] = (0.5, 0.4, 0.0)
    
    transform = compute_ring_transform(landmarks, finger='ring', image_shape=(100, 100))
    
    assert transform is not None
    assert transform['x'] == 50
    assert transform['y'] == 45 # Midpoint of 50 and 40
    # Vector is (0, -10). Angle is -90 deg.
    # Rotation = -90 + 90 = 0.
    assert transform['rotation'] == 0.0
    assert transform['scale'] > 0

def test_compute_ring_transform_invalid_finger():
    landmarks = [(0.5, 0.5, 0.0)] * 21
    transform = compute_ring_transform(landmarks, finger='invalid')
    assert transform is None
