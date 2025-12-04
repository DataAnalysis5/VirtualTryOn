# Virtual Ring Try-On

A production-ready Python application for virtual ring try-on using MediaPipe Hands and OpenCV. This project supports both real-time webcam feed and static image processing.

## Features

- **Real-time Webcam Mode**: Detects hands and overlays rings on selected fingers.
- **Static Image Mode**: Upload an image or use CLI to process local files.
- **Accurate Placement**: Uses hand landmarks to estimate finger orientation, scale, and position.
- **Occlusion Handling**: Basic handling using landmark depth.
- **Shading**: Adjusts ring brightness to match the environment.
- **Custom Assets**: Easily add your own transparent PNG ring images.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd VirtualTryOn
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit App (GUI)

Run the interactive web application:

```bash
streamlit run src/app.py
```

- **Webcam Mode**: Allow camera access and see the ring on your finger in real-time.
- **Upload Mode**: Upload a photo of a hand to try on rings.
- **Controls**: Use the sidebar to select the ring, finger, and adjust scale/rotation.

### CLI (Command Line Interface)

Process a single image:

```bash
python src/app.py --image examples/demo_image_1.jpg --ring ring1.png --finger ring --output result.jpg
```

Arguments:
- `--image`: Path to input image.
- `--ring`: Name of the ring asset (must be in `examples/rings` or specified assets dir).
- `--finger`: Target finger (`index`, `middle`, `ring`, `pinky`, `thumb`).
- `--output`: Path to save the result.

## Docker

Build and run with Docker:

```bash
docker build -t virtual-try-on .
docker run -p 8501:8501 virtual-try-on
```

Access the app at `http://localhost:8501`.

## Project Structure

- `src/`: Source code.
  - `app.py`: Main entry point.
  - `detector.py`: MediaPipe wrapper.
  - `ring_fitter.py`: Logic for placing the ring.
  - `compositor.py`: Image overlay and blending.
- `examples/`: Sample images and ring assets.
- `tests/`: Unit tests.
- `docs/`: Detailed documentation.

## How it Works

1. **Detection**: MediaPipe Hands detects 21 landmarks on the hand.
2. **Fitting**: 
   - The ring is placed on the proximal phalanx (between MCP and PIP joints).
   - **Position**: Midpoint of MCP and PIP.
   - **Rotation**: Calculated from the vector pointing from MCP to PIP.
   - **Scale**: Estimated based on the distance between MCP and PIP.
3. **Compositing**: The ring image is scaled, rotated, and overlaid. Brightness is adjusted based on the background area.

## Troubleshooting

- **No hand detected**: Ensure the hand is clearly visible and lighting is good.
- **Ring misalignment**: Use the sidebar sliders to fine-tune the position and scale.
- **Webcam not working**: Check browser permissions or try a different browser.

## License

MIT
