import pytest
import numpy as np
from src.compositor import overlay_ring

def test_overlay_ring_basic():
    # Background: 100x100 black
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    # Ring: 10x10 white square
    ring = np.ones((10, 10, 3), dtype=np.uint8) * 255
    
    transform = {
        'x': 50,
        'y': 50,
        'scale': 20.0, # Target width 20
        'rotation': 0
    }
    
    result = overlay_ring(bg, ring, transform, apply_shading=False)
    
    # Center pixel should be white (or close to it)
    assert np.array_equal(result[50, 50], [255, 255, 255])
    # Corner pixel should be black
    assert np.array_equal(result[0, 0], [0, 0, 0])

def test_overlay_ring_out_of_bounds():
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    ring = np.ones((10, 10, 3), dtype=np.uint8) * 255
    
    transform = {
        'x': 200, # Way out
        'y': 200,
        'scale': 20.0,
        'rotation': 0
    }
    
    result = overlay_ring(bg, ring, transform)
    assert np.array_equal(result, bg)
