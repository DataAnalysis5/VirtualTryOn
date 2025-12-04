# Architecture

## Overview

The application follows a modular design with separation of concerns:

1. **Input Layer**: Handles image acquisition (Webcam or File).
2. **Detection Layer**: Uses MediaPipe to extract semantic information (Landmarks).
3. **Logic Layer**: Computes geometric transforms based on landmarks.
4. **Rendering Layer**: Composites the virtual object onto the image.
5. **Presentation Layer**: Streamlit UI or CLI.

## Modules

### `detector.py`
Wraps `mediapipe.solutions.hands`.
- **Input**: Image (BGR).
- **Output**: List of landmarks (normalized), Bounding Boxes.
- **Config**: Confidence thresholds, max hands.

### `ring_fitter.py`
Pure logic component.
- **Input**: Landmarks, Finger selection.
- **Output**: Transform dict (x, y, scale, rotation, z_depth).
- **Math**:
  - **Position**: $\frac{P_{mcp} + P_{pip}}{2}$
  - **Rotation**: $\text{atan2}(P_{pip}.y - P_{mcp}.y, P_{pip}.x - P_{mcp}.x)$
  - **Scale**: $\|P_{pip} - P_{mcp}\| \times k$

### `compositor.py`
Image processing component.
- **Input**: Background image, Ring asset, Transform.
- **Output**: Composited image.
- **Features**:
  - Affine transformations (Scale, Rotate, Translate).
  - Alpha blending.
  - Brightness matching (V-channel adjustment in HSV).

### `assets_manager.py`
Resource management.
- Loads and caches PNG images.
- Handles directory scanning.

## Data Flow

```
[Image Source] -> [Detector] -> [Landmarks]
                                     |
                                     v
[Ring Asset] -> [Ring Fitter] -> [Transform]
                                     |
                                     v
[Image Source] + [Ring Asset] -> [Compositor] -> [Output Image]
```
