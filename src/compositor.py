import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def adjust_brightness(ring_img, bg_roi):
    """
    Adjusts the brightness of the ring image to match the background ROI.
    """
    # Convert to HSV
    if ring_img.shape[2] == 4:
        ring_rgb = ring_img[:, :, :3]
        alpha = ring_img[:, :, 3]
    else:
        ring_rgb = ring_img
        alpha = None

    hsv_ring = cv2.cvtColor(ring_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_bg = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Average V
    v_ring = np.mean(hsv_ring[:, :, 2])
    v_bg = np.mean(hsv_bg[:, :, 2])

    if v_ring == 0: return ring_img

    # Scale factor
    # Don't make it too dark or too bright.
    ratio = v_bg / v_ring
    # Clamp ratio
    ratio = np.clip(ratio, 0.6, 1.4)

    hsv_ring[:, :, 2] = np.clip(hsv_ring[:, :, 2] * ratio, 0, 255)

    ring_rgb_adj = cv2.cvtColor(hsv_ring.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if alpha is not None:
        return np.dstack([ring_rgb_adj, alpha])
    else:
        return ring_rgb_adj

def overlay_ring(image, ring_image, transform, apply_shading=True, mask_side=None):
    """
    Overlays the ring image onto the background image using the transform.
    
    Args:
        image: Background image (BGR).
        ring_image: Ring image (BGRA).
        transform: Dict with x, y, scale, rotation.
        apply_shading: Boolean.
        mask_side: 'top' or 'bottom' or None. Masks the vertical center strip of that half.
        
    Returns:
        Composited image.
    """
    if ring_image is None or transform is None:
        return image

    h_bg, w_bg = image.shape[:2]
    
    # 1. Scale
    scale = transform['scale']
    h_ring, w_ring = ring_image.shape[:2]
    if w_ring == 0 or h_ring == 0: return image
    
    scale_factor = scale / w_ring
    new_w = int(w_ring * scale_factor)
    new_h = int(h_ring * scale_factor)
    
    if new_w <= 0 or new_h <= 0: return image
    
    resized_ring = cv2.resize(ring_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Apply Masking (Occlusion)
    if mask_side:
        if resized_ring.shape[2] == 3:
            resized_ring = cv2.cvtColor(resized_ring, cv2.COLOR_BGR2BGRA)
            
        # Define strip width (approx finger width relative to ring)
        # Ring scale is roughly finger width.
        # Let's mask 60% of the center.
        strip_w = int(new_w * 0.6)
        strip_x_start = (new_w - strip_w) // 2
        strip_x_end = strip_x_start + strip_w
        
        # Soft mask or Hard mask? Hard for now.
        if mask_side == 'top':
            # Mask top half (Band) - for Back of Hand view
            resized_ring[0:new_h//2, strip_x_start:strip_x_end, 3] = 0
        elif mask_side == 'bottom':
            # Mask bottom half (Gem) - for Palm view
            resized_ring[new_h//2:, strip_x_start:strip_x_end, 3] = 0
    
    # 2. Rotate
    
    # 2. Rotate
    # Rotate around center
    angle = transform['rotation']
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box to avoid clipping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w_rot = int((new_h * sin) + (new_w * cos))
    new_h_rot = int((new_h * cos) + (new_w * sin))
    
    M[0, 2] += (new_w_rot / 2) - center[0]
    M[1, 2] += (new_h_rot / 2) - center[1]
    
    rotated_ring = cv2.warpAffine(resized_ring, M, (new_w_rot, new_h_rot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # 3. Place
    # transform['x'], transform['y'] is the center position on background
    c_x, c_y = transform['x'], transform['y']
    
    # Top-left corner
    tl_x = c_x - new_w_rot // 2
    tl_y = c_y - new_h_rot // 2
    
    # Clip to background
    x1 = max(0, tl_x)
    y1 = max(0, tl_y)
    x2 = min(w_bg, tl_x + new_w_rot)
    y2 = min(h_bg, tl_y + new_h_rot)
    
    # ROI in ring image
    r_x1 = x1 - tl_x
    r_y1 = y1 - tl_y
    r_x2 = r_x1 + (x2 - x1)
    r_y2 = r_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return image
        
    ring_roi = rotated_ring[r_y1:r_y2, r_x1:r_x2]
    bg_roi = image[y1:y2, x1:x2]
    
    # Shading
    if apply_shading:
        ring_roi = adjust_brightness(ring_roi, bg_roi)
    
    # Blend
    # Extract alpha
    if ring_roi.shape[2] == 4:
        alpha = ring_roi[:, :, 3] / 255.0
        ring_rgb = ring_roi[:, :, :3]
    else:
        alpha = np.ones((ring_roi.shape[0], ring_roi.shape[1]))
        ring_rgb = ring_roi
        
    # Expand alpha to 3 channels
    alpha_3 = np.dstack([alpha, alpha, alpha])
    
    # Composite
    # out = src * alpha + dst * (1 - alpha)
    composited = (ring_rgb * alpha_3 + bg_roi * (1.0 - alpha_3)).astype(np.uint8)
    
    image_copy = image.copy()
    image_copy[y1:y2, x1:x2] = composited
    
    return image_copy
