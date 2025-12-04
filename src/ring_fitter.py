import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# MediaPipe Landmark Indices
FINGER_INDICES = {
    'thumb': [1, 2, 3, 4], # CMC, MCP, IP, TIP
    'index': [5, 6, 7, 8], # MCP, PIP, DIP, TIP
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

def compute_ring_transform(landmarks, finger='ring', image_shape=(480, 640)):
    """
    Computes the transform (position, scale, rotation) for the ring.
    
    Args:
        landmarks: List of (x, y, z) normalized coordinates.
        finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'.
        image_shape: (height, width) of the image.
        
    Returns:
        dict: {
            'x': int, 'y': int, # Center position
            'scale': float,     # Scale factor for the ring image
            'rotation': float,  # Rotation in degrees
            'z_depth': float    # Depth hint
        }
    """
    if finger not in FINGER_INDICES:
        logger.error(f"Invalid finger name: {finger}")
        return None

    indices = FINGER_INDICES[finger]
    
    # For most fingers, the ring sits on the proximal phalanx (between MCP and PIP).
    # For thumb, it's between MCP and IP.
    # Indices: [MCP, PIP, ...] (except thumb which is [CMC, MCP, IP, TIP] - wait, thumb indices are 1,2,3,4. 1 is CMC, 2 is MCP, 3 is IP, 4 is TIP. Ring usually sits on proximal phalanx which is between MCP(2) and IP(3)?)
    # Let's assume for thumb it's between 2 and 3.
    # For others, it's between indices[0] (MCP) and indices[1] (PIP).
    
    if finger == 'thumb':
        idx_base = indices[1] # MCP
        idx_tip = indices[2]  # IP
    else:
        idx_base = indices[0] # MCP
        idx_tip = indices[1]  # PIP

    # Get coordinates
    h, w = image_shape[:2]
    
    # Base (MCP)
    p1 = np.array([landmarks[idx_base][0] * w, landmarks[idx_base][1] * h])
    # Tip (PIP/IP)
    p2 = np.array([landmarks[idx_tip][0] * w, landmarks[idx_tip][1] * h])
    
    # Midpoint for placement
    center = (p1 + p2) / 2
    
    # Vector
    vec = p2 - p1
    length = np.linalg.norm(vec)
    
    # Rotation
    # Calculate angle with horizontal axis
    # atan2(y, x) gives angle in radians. 
    # We want the rotation of the ring image. Assuming ring image is vertical (up-down) or horizontal?
    # Usually ring images are viewed from front (circle). If we place it on finger, we want it to align with finger.
    # If the ring image is a circle (front view), rotation might not matter much unless it has a gem.
    # If the ring image implies a perspective (e.g. oval), we need to align the major/minor axis.
    # Let's assume the ring asset is a "front view" or "top view" that needs to be rotated to match finger direction.
    # If the ring image is upright (0 degrees), and finger is pointing up (-90 degrees in image coords?), we need to adjust.
    # Let's assume standard ring asset is oriented such that "up" is the direction of the finger.
    # So if finger points right (0 deg), ring should be rotated -90? 
    # Let's stick to: Ring image 'up' corresponds to Finger 'distal' direction.
    
    angle_rad = math.atan2(vec[1], vec[0])
    angle_deg = math.degrees(angle_rad)
    
    # Correction: atan2(dy, dx). If finger points right (1, 0), angle is 0. 
    # If ring image is upright (vertical), we need to rotate it -90 to point right.
    # Let's assume ring image is upright.
    rotation = angle_deg + 90 

    # Scale
    # Heuristic: Ring width is roughly proportional to phalanx length.
    # Let's say ring width should be about 0.4 * phalanx length (this is a guess, tuneable).
    # Or we can use the width of the finger if we had it.
    # Distance between MCP and PIP is the length of the proximal phalanx.
    # A ring diameter is usually slightly wider than the finger width.
    # Finger width is approx 1/3 to 1/2 of proximal phalanx length?
    # Let's start with scale = length * 0.5
    scale_factor = length * 0.8 # Tunable parameter
    
    # Depth
    z_depth = landmarks[idx_base][2]

    return {
        'x': int(center[0]),
        'y': int(center[1]),
        'scale': scale_factor,
        'rotation': rotation,
        'z_depth': z_depth
    }
