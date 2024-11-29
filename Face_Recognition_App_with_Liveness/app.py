import streamlit as st
import cv2 as cv
import face_recognition
import dlib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from imutils import face_utils
from scipy.spatial import distance as dist
import time
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Attendance System",
    page_icon="📸",
    layout="wide"
)

# Add custom CSS to control video container width
st.markdown("""
    <style>
    .stVideo {
        width: 50%;
        margin: auto;
    }
    .stImage {
        max-width: 160px !important;
        margin: auto;
    }
    .stImage > img {
        width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
known_face_path = st.sidebar.file_uploader("Upload Reference Image", type=['png', 'jpg', 'jpeg'])
start_button = st.sidebar.button("Start Session")
stop_button = st.sidebar.button("Stop Session")

# Main content
st.title("AI Enhanced Engagement Tracker with Attendance System")
st.write("This system tracks attention and liveness during attendance.")

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'Name', 'Date', 'Time', 'Screenshot', 'Attentive', 'Attention Score', 'Liveness Score'
    ])
if 'attention_scores' not in st.session_state:
    st.session_state.attention_scores = []
if 'liveness_scores' not in st.session_state:
    st.session_state.liveness_scores = []
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = None

# Constants and initializations
MAX_YAW_THRESHOLD = 0.5
MAX_PITCH_THRESHOLD = 0.5
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.5

# Initialize dlib's face detector and predictor
@st.cache_resource
def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, landmark_predictor = load_face_detector()

# Create screenshots directory
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Helper functions (copy all the helper functions from avg_atten_live_score.py)
def get_head_pose(landmarks):
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-165.0, 170.0, -135.0),     # Left eye left corner
        (165.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    focal_length = 320
    center = (160, 120)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    return rotation_vector, translation_vector


def calculate_attention_score(yaw, pitch):
    yaw_score = max(0, 1 - abs(yaw[0]) / MAX_YAW_THRESHOLD)
    pitch_score = max(0, 1 - abs(pitch[0]) / MAX_PITCH_THRESHOLD)
    return (yaw_score + pitch_score) / 2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[3], mouth[5])
    C = dist.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)

def calculate_basic_liveness_score(ear, mar, last_blink_time, last_mouth_movement_time):
    current_time = time.time()
    blink_score = min(1.0, 1.0 / (current_time - last_blink_time + 1))
    mouth_score = min(1.0, 1.0 / (current_time - last_mouth_movement_time + 1))
    return (blink_score + mouth_score) / 2


# Main app functionality
def run_attendance_system():
    # Create placeholders for video feed and metrics
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize camera
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        st.error("Camera not working")
        return

    # Initialize counters
    COUNTER = 0
    TOTAL = 0
    last_blink_time = time.time()
    last_mouth_movement_time = time.time()
    frame_count = 0

    try:
        while st.session_state.running:
            frame_count += 1
            ret, frame = cam.read()
            
            if not ret:
                st.error("Can't receive frame")
                break

            if frame_count % 2 == 0:
                continue

            # Resize frame to smaller dimensions
            frame = cv.resize(frame, (160, 120))  # Make the initial frame smaller
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Face detection and recognition logic
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations and st.session_state.known_faces is not None:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    distance = face_recognition.face_distance([st.session_state.known_faces], face_encoding)[0]

                    if distance < 0.6:
                        now = datetime.now()
                        name = 'Manas'
                        
                        face_landmarks = landmark_predictor(gray, dlib.rectangle(left, top, right, bottom))
                        shape = face_utils.shape_to_np(face_landmarks)
                        landmarks = [(p.x, p.y) for p in face_landmarks.parts()]
                        
                        # Calculate attention score
                        rotation_vector, translation_vector = get_head_pose(landmarks)
                        yaw, pitch, roll = rotation_vector
                        attention_score = calculate_attention_score(yaw, pitch)
                        st.session_state.attention_scores.append(attention_score)
                        
                        # Calculate liveness components
                        leftEye = shape[42:48]
                        rightEye = shape[36:42]
                        mouth = shape[60:68]
                        
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        mar = mouth_aspect_ratio(mouth)
                        
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1
                        else:
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                                last_blink_time = time.time()
                            COUNTER = 0
                        
                        if mar > MOUTH_AR_THRESH:
                            last_mouth_movement_time = time.time()
                        
                        liveness_score = calculate_basic_liveness_score(
                            ear, mar, last_blink_time, last_mouth_movement_time
                        )
                        st.session_state.liveness_scores.append(liveness_score)
                        
                        # Update DataFrame
                        screenshot_filename = f"screenshots/{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                        cv.imwrite(screenshot_filename, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                        
                        new_entry = pd.DataFrame({
                            'Name': [name],
                            'Date': [now.strftime("%Y-%m-%d")],
                            'Time': [now.strftime("%H:%M:%S")],
                            'Screenshot': [screenshot_filename],
                            'Attentive': ['Yes' if attention_score >= 0.4 else 'No'],
                            'Attention Score': [attention_score],
                            'Liveness Score': [liveness_score]
                        })
                        st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)

                        # Draw on frame
                        cv.rectangle(frame, (left, top), (right, bottom), 
                                   (0, 255, 0) if attention_score >= 0.4 else (0, 0, 255), 2)
                        
                        # Update metrics display
                        with metrics_placeholder.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Attention Score", f"{attention_score:.2f}")
                            with col2:
                                st.metric("Liveness Score", f"{liveness_score:.2f}")

            # Display the frame in a smaller container
            col1, col2, col3 = st.columns([2,1,2])  # Adjust column ratios to make middle column smaller
            with col2:  # Use middle column for video
                video_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        cam.release()
        # Calculate and display final results
        if st.session_state.attention_scores:
            average_attention_score = np.mean(st.session_state.attention_scores)
            average_liveness_score = np.mean(st.session_state.liveness_scores)
            
            st.success("Session Complete!")
            st.write(f"Average Attention Score: {average_attention_score:.2f}")
            st.write(f"Average Liveness Score: {average_liveness_score:.2f}")
            
            # Save results to Excel
            if not st.session_state.df.empty:
                avg_entry = pd.DataFrame({
                    'Name': ['Average'],
                    'Date': [''],
                    'Time': [''],
                    'Screenshot': [''],
                    'Attentive': [''],
                    'Attention Score': [average_attention_score],
                    'Liveness Score': [average_liveness_score]
                })
                final_df = pd.concat([st.session_state.df, avg_entry], ignore_index=True)
                final_df.to_excel('attendance_log.xlsx', index=False)
                
                # Provide download button for the Excel file
                with open('attendance_log.xlsx', 'rb') as f:
                    st.download_button(
                        label="Download Attendance Report",
                        data=f,
                        file_name='attendance_log.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

# Handle start/stop actions
if start_button:
    if known_face_path is None:
        st.warning("Please upload a reference image first!")
    else:
        # Convert uploaded file to numpy array
        file_bytes = np.asarray(bytearray(known_face_path.read()), dtype=np.uint8)
        known_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        known_image_rgb = cv.cvtColor(known_image, cv.COLOR_BGR2RGB)
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(known_image_rgb)
        if len(face_encodings) == 0:
            st.error("No face found in the reference image!")
        else:
            st.session_state.known_faces = face_encodings[0]
            st.session_state.running = True
            run_attendance_system()

if stop_button:
    st.session_state.running = False
    
    # Calculate and display final results if there are scores
    if st.session_state.attention_scores:
        average_attention_score = np.mean(st.session_state.attention_scores)
        
        # Create a results section with prominent display
        st.markdown("---")  # Add a separator
        st.header("Session Results")
        
        # Display attention score with colored background
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: {'#90EE90' if average_attention_score >= 0.4 else '#FFB6C1'};
                text-align: center;
                max-width: 500px;
                margin: auto;
            ">
                <h3>Average Attention Score</h3>
                <h2>{average_attention_score:.2f}</h2>
                <p>{'Attentive' if average_attention_score >= 0.4 else 'Not Attentive'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display final attendance status
        is_present = average_attention_score >= 0.4
        st.markdown(
            f"""
            <div style="
                margin-top: 20px;
                padding: 20px;
                border-radius: 10px;
                background-color: {'#90EE90' if is_present else '#FFB6C1'};
                text-align: center;
                max-width: 500px;
                margin: auto;
            ">
                <h2>Final Attendance Status: {' PRESENT' if is_present else ' ABSENT'}</h2>
                {f'<p>Reason: Low attention</p>' if not is_present else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

# Display current session data
if not st.session_state.df.empty:
    st.write("Current Session Data:")
    st.dataframe(st.session_state.df) 