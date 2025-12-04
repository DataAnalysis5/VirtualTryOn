import sys
import os
import argparse
import cv2
import numpy as np
import logging
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import HandDetector
from src.ring_fitter import compute_ring_transform
from src.compositor import overlay_ring
from src.assets_manager import AssetsManager
from src.utils import setup_logging, cv2_to_pil, pil_to_cv2

setup_logging()
logger = logging.getLogger(__name__)

def run_cli(args):
    """Runs the application in CLI mode for static images."""
    logger.info(f"Running in CLI mode. Image: {args.image}")
    
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return

    # Load resources
    assets_mgr = AssetsManager(args.assets_dir)
    detector = HandDetector(static_image_mode=True, max_num_hands=1)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        logger.error("Failed to read image.")
        return

    # Detect
    results, landmarks_list, _, handedness_list = detector.detect(image)
    
    if not landmarks_list:
        logger.warning("No hands detected.")
        return

    # Process
    ring_img = assets_mgr.get_ring(args.ring)
    if ring_img is None:
        logger.error(f"Ring asset not found: {args.ring}")
        # List available
        logger.info(f"Available rings: {assets_mgr.get_available_rings()}")
        return

    # Fit and Overlay
    # Use first hand
    landmarks = landmarks_list[0]
    handedness = handedness_list[0]
    
    transform = compute_ring_transform(landmarks, finger=args.finger, image_shape=image.shape)
    
    if transform:
        is_palm = detector.is_palm_facing(landmarks, handedness)
        mask_side = 'bottom' if is_palm else 'top'
        logger.info(f"Handedness: {handedness}, Palm Facing: {is_palm}, Mask Side: {mask_side}")
        
        # Apply manual adjustments if any (CLI could support them but keeping simple)
        final_image = overlay_ring(image, ring_img, transform, apply_shading=True, mask_side=mask_side)
        
        output_path = args.output
        cv2.imwrite(output_path, final_image)
        logger.info(f"Saved output to {output_path}")
    else:
        logger.error("Could not compute ring transform.")

def run_streamlit():
    import streamlit as st
    
    st.set_page_config(page_title="Virtual Ring Try-On", layout="wide")
    
    st.title("💍 Virtual Ring Try-On")
    st.markdown("Try on rings in real-time using your webcam or upload an image.")

    # Sidebar
    st.sidebar.header("Settings")
    
    # Mode
    mode = st.sidebar.radio("Mode", ["Webcam", "Upload Image"])
    
    # Assets
    assets_dir = os.path.join(os.path.dirname(__file__), '../examples/rings')
    assets_mgr = AssetsManager(assets_dir)
    available_rings = assets_mgr.get_available_rings()
    
    if not available_rings:
        st.sidebar.error("No ring assets found in examples/rings!")
        return

    selected_ring_name = st.sidebar.selectbox("Select Ring", available_rings)
    ring_img = assets_mgr.get_ring(selected_ring_name)
    
    # Finger Selection
    finger = st.sidebar.selectbox("Select Finger", ['index', 'middle', 'ring', 'pinky', 'thumb'], index=2)
    
    # Adjustments
    st.sidebar.subheader("Adjustments")
    scale_adj = st.sidebar.slider("Scale Adjustment", 0.5, 1.5, 1.0, 0.05)
    rot_adj = st.sidebar.slider("Rotation Adjustment", -45, 45, 0, 1)
    y_adj = st.sidebar.slider("Vertical Offset", -50, 50, 0, 1)
    
    apply_shading = st.sidebar.checkbox("Apply Shading", value=True)
    apply_occlusion = st.sidebar.checkbox("Apply Occlusion", value=True)
    
    # Detector
    detector = HandDetector(static_image_mode=(mode == "Upload Image"), min_detection_confidence=0.5)

    if mode == "Webcam":
        run_webcam_mode(detector, ring_img, finger, scale_adj, rot_adj, y_adj, apply_shading, apply_occlusion)
    else:
        run_upload_mode(detector, ring_img, finger, scale_adj, rot_adj, y_adj, apply_shading, apply_occlusion)

def run_webcam_mode(detector, ring_img, finger, scale_adj, rot_adj, y_adj, apply_shading, apply_occlusion):
    import streamlit as st
    
    # Use streamlit-webrtc for better performance? 
    # For simplicity and standard requirements, use st.camera_input or cv2 loop.
    # st.camera_input returns a static image snapshot, not a live stream for processing.
    # To do live processing in Streamlit, we usually use a loop with an empty placeholder.
    
    run_live = st.checkbox("Start Webcam", value=False)
    
    if run_live:
        st.info("Starting webcam... Press 'Stop' to end.")
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        
        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            # Detect
            results, landmarks_list, _, handedness_list = detector.detect(frame)
            
            output_frame = frame.copy()
            
            if landmarks_list:
                # Use first hand
                landmarks = landmarks_list[0]
                handedness = handedness_list[0]
                
                # Draw landmarks (optional debug)
                # detector.draw_landmarks(output_frame, results)
                
                transform = compute_ring_transform(landmarks, finger=finger, image_shape=frame.shape)
                
                if transform:
                    # Apply adjustments
                    transform['scale'] *= scale_adj
                    transform['rotation'] += rot_adj
                    transform['y'] += y_adj
                    
                    mask_side = None
                    if apply_occlusion:
                        is_palm = detector.is_palm_facing(landmarks, handedness)
                        mask_side = 'bottom' if is_palm else 'top'
                    
                    output_frame = overlay_ring(output_frame, ring_img, transform, apply_shading=apply_shading, mask_side=mask_side)
            
            # Display
            st_frame.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
        cap.release()

def run_upload_mode(detector, ring_img, finger, scale_adj, rot_adj, y_adj, apply_shading, apply_occlusion):
    import streamlit as st
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        results, landmarks_list, _, handedness_list = detector.detect(image_bgr)
        
        output_image = image_bgr.copy()
        
        if landmarks_list:
            landmarks = landmarks_list[0]
            handedness = handedness_list[0]
            
            transform = compute_ring_transform(landmarks, finger=finger, image_shape=image_bgr.shape)
            
            if transform:
                transform['scale'] *= scale_adj
                transform['rotation'] += rot_adj
                transform['y'] += y_adj
                
                mask_side = None
                if apply_occlusion:
                    is_palm = detector.is_palm_facing(landmarks, handedness)
                    mask_side = 'bottom' if is_palm else 'top'
                
                output_image = overlay_ring(output_image, ring_img, transform, apply_shading=apply_shading, mask_side=mask_side)
            else:
                st.warning("Could not compute transform.")
        else:
            st.warning("No hand detected.")
            
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Result")

if __name__ == "__main__":
    # Check if running via streamlit
    # Streamlit sets 'streamlit' in modules or specific env vars?
    # Reliable way: check if we are being run by `streamlit run`
    # But we can also check arguments.
    
    try:
        import streamlit.runtime
        is_streamlit = streamlit.runtime.exists()
    except ImportError:
        # Fallback for older streamlit
        is_streamlit = False
        
    # If running directly with python, sys.argv might contain arguments for CLI
    # If running with streamlit, sys.argv contains 'streamlit', 'run', ...
    
    if len(sys.argv) > 1 and sys.argv[0].endswith('streamlit'):
         # This branch might not be hit because streamlit executes the script content.
         pass

    # Simple heuristic: if --image is in args, it's CLI.
    if '--image' in sys.argv:
        parser = argparse.ArgumentParser(description="Virtual Ring Try-On CLI")
        parser.add_argument('--image', type=str, required=True, help="Path to input image")
        parser.add_argument('--ring', type=str, required=True, help="Name of ring asset (e.g. ring1.png)")
        parser.add_argument('--finger', type=str, default='ring', help="Finger to place ring on")
        parser.add_argument('--assets_dir', type=str, default='examples/rings', help="Directory of ring assets")
        parser.add_argument('--output', type=str, default='output.jpg', help="Output path")
        
        args = parser.parse_args()
        run_cli(args)
    else:
        # Run Streamlit
        # If we are here, it means we are either running `python app.py` (no args) or `streamlit run app.py`
        # If `python app.py`, we should probably tell user to run with streamlit or provide args.
        # But to satisfy the requirement "app.py (Streamlit app entrypoint)", we can just run the function.
        # If run with python, it won't render streamlit UI.
        
        # We can detect if we are inside streamlit loop.
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx():
                run_streamlit()
            else:
                print("Please run with: streamlit run src/app.py")
                print("Or use CLI mode: python src/app.py --image ...")
        except ImportError:
             # Fallback
             run_streamlit()
