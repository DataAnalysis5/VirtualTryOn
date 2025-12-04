import logging
import cv2
import numpy as np
from PIL import Image

def setup_logging(level=logging.INFO):
    """Configures the logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR image to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL RGB image to an OpenCV BGR image."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_image_keep_aspect(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resizes an image maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image
