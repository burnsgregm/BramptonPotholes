
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np

# --- 1. Global Configurations ---
APP_TITLE = "Brampton Pothole Detector (PoC)"
MODEL_FILE = "pothole_model.pt" # The trained model we will deploy

# --- 2. Caching & Model Loading ---

@st.cache_resource
def load_yolo_model():
    """Loads the custom-trained YOLOv8 model."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file not found: {MODEL_FILE}. Please ensure it's in the repo.")
        st.stop()
    
    try:
        model = YOLO(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

def process_image(model, image):
    """Processes a single image, runs inference, and returns the plotted image."""
    results = model(image)
    plotted_image = results[0].plot() # .plot() draws boxes on the image
    return plotted_image

def process_video(model, video_file_path):
    """Processes a video, runs inference frame-by-frame, and returns the path to the new video."""
    
    # Create a temporary file to save the processed video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = tfile.name

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pothole_log = []
    frame_count = 0
    
    progress_bar = st.progress(0, text="Processing video, please wait...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        plotted_frame = results[0].plot()
        
        # Log detections
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.conf[0] > 0.6: # Confidence threshold
                    pothole_log.append(f"[Frame {frame_count:04d}] Pothole Detected.")

        # Write the processed frame
        out.write(plotted_frame)
        
        # Update progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            progress_bar.progress(frame_count / total_frames, text=f"Processing frame {frame_count}/{total_frames}...")
        else:
            progress_bar.progress(frame_count / 1000, text=f"Processing frame {frame_count}...") # Fallback for unknown length

    cap.release()
    out.release()
    progress_bar.empty()
    
    return output_video_path, pothole_log

# --- 3. Streamlit App UI ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🤖 {APP_TITLE}")
st.markdown("This prototype is a demonstration for the **City of Brampton AI PoC Program**. Upload an image or a short video of a road to detect potholes.")

# Load the model
model = load_yolo_model()

# Create tabs for Image and Video
tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎬 Video Detection"])

with tab1:
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        with st.spinner("Analyzing image..."):
            plotted_image = process_image(model, image)
            
        st.image(plotted_image, caption="Processed Image with Pothole Detections", use_column_width=True)

with tab2:
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name) # Show the original video
        
        if st.button("Detect Potholes in Video", use_container_width=True):
            output_video_path, pothole_log = process_video(model, tfile.name)
            
            if output_video_path:
                st.subheader("Processed Video")
                st.video(output_video_path)
                
                st.subheader("Live Asset Management Log")
                st.code('\n'.join(pothole_log[-20:])) # Show last 20 detections
                
                # Provide a download link for the processed video
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="pothole_detection_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            # Clean up temp file
            os.remove(tfile.name)
