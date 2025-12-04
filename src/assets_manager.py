import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AssetsManager:
    def __init__(self, assets_dir: str):
        self.assets_dir = assets_dir
        self.rings = {}
        self._load_assets()

    def _load_assets(self):
        """Loads all PNG images from the assets directory."""
        if not os.path.exists(self.assets_dir):
            logger.warning(f"Assets directory not found: {self.assets_dir}")
            os.makedirs(self.assets_dir, exist_ok=True)
            return

        for filename in os.listdir(self.assets_dir):
            if filename.lower().endswith('.png'):
                path = os.path.join(self.assets_dir, filename)
                # Read with alpha channel
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    self.rings[filename] = image
                    logger.info(f"Loaded asset: {filename}")
                else:
                    logger.error(f"Failed to load asset: {path}")

    def get_ring(self, name: str) -> np.ndarray:
        """Returns the ring image by name."""
        return self.rings.get(name)

    def get_available_rings(self) -> list:
        """Returns a list of available ring names."""
        return list(self.rings.keys())
