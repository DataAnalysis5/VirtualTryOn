# Usage Guide

## Adding New Rings

1. Prepare your ring image.
   - Format: PNG with transparent background.
   - Orientation: The ring should be upright (top of the ring pointing up).
   - Crop: Crop closely to the ring edges.
2. Place the PNG file in `examples/rings/`.
3. Restart the Streamlit app or run the CLI. The new ring will be automatically detected.

## Tuning Placement

If the automatic placement is consistently off for a specific ring asset (e.g., it's too wide), you might need to adjust the code or use the UI sliders.

- **Scale**: If the ring looks too big, reduce the scale slider.
- **Rotation**: If the ring is tilted, adjust the rotation slider.
- **Vertical Offset**: If the ring sits too high or low on the finger, adjust the vertical offset.

## CLI Advanced Usage

You can specify a custom assets directory:

```bash
python src/app.py --image my_hand.jpg --ring custom_ring.png --assets_dir /path/to/my/rings
```

## Performance

- The application runs on CPU.
- MediaPipe is optimized for mobile and desktop CPUs.
- For smoother webcam performance, ensure good lighting so the detector doesn't struggle.
