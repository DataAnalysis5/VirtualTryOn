# API Reference

## `src.detector`

### `HandDetector`

```python
class HandDetector(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
```

- **detect(image)**: Returns `(results, landmarks_list, bboxes)`.

## `src.ring_fitter`

### `compute_ring_transform`

```python
def compute_ring_transform(landmarks, finger='ring', image_shape=(480, 640)) -> dict
```

Calculates the placement of the ring.
- **landmarks**: List of (x, y, z) tuples.
- **finger**: 'thumb', 'index', 'middle', 'ring', 'pinky'.
- **Returns**: Dictionary with keys `x`, `y`, `scale`, `rotation`, `z_depth`.

## `src.compositor`

### `overlay_ring`

```python
def overlay_ring(image, ring_image, transform, apply_shading=True) -> np.ndarray
```

Composites the ring onto the image.

### `adjust_brightness`

```python
def adjust_brightness(ring_img, bg_roi) -> np.ndarray
```

Matches the V-channel mean of the ring to the background ROI.
