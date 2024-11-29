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

# Initialize dlib's face detector and the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# Create a directory to save screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Load the known image
known_image = face_recognition.load_image_file("My_image.png")
known_faces = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# Create a DataFrame to store recognized face information
columns = ['Name', 'Date', 'Time', 'Screenshot', 'Attentive', 'Attention Score', 'Liveness Score']
df = pd.DataFrame(columns=columns)

# Initialize lists for averaging scores
attention_scores = []
liveness_scores = []

# Launch the live camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working")
    exit()

# Define thresholds
MAX_YAW_THRESHOLD = 0.5
MAX_PITCH_THRESHOLD = 0.5
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.5

# Initialize counters
COUNTER = 0
TOTAL = 0
last_blink_time = time.time()
last_mouth_movement_time = time.time()

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

frame_count = 0
try:
    while True:
        frame_count += 1
        ret, frame = cam.read()
        
        if not ret:
            print("Can't receive frame")
            break

        if frame_count % 2 == 0:
            continue

        frame = cv.resize(frame, (320, 240))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            continue

        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distance = face_recognition.face_distance([known_faces], face_encoding)[0]

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
                attention_scores.append(attention_score)
                
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
                liveness_scores.append(liveness_score)
                
                attentive = 'Yes' if attention_score >= 0.4 else 'No'
                screenshot_filename = f"screenshots/{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

                # Display scores
                if attentive == 'Yes':
                    cv.putText(frame, f'Attentive (Score: {attention_score:.2f})', (10, 30), 
                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                else:
                    cv.putText(frame, f'Not Attentive (Score: {attention_score:.2f})', (10, 30), 
                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                
                cv.putText(frame, f'Liveness: {liveness_score:.2f}', (10, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

                # Draw landmarks and save data
                for (x, y) in landmarks:
                    cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

                cv.imwrite(screenshot_filename, frame)
                new_entry = pd.DataFrame({
                    'Name': [name],
                    'Date': [now.strftime("%Y-%m-%d")],
                    'Time': [now.strftime("%H:%M:%S")],
                    'Screenshot': [screenshot_filename],
                    'Attentive': [attentive],
                    'Attention Score': [attention_score],
                    'Liveness Score': [liveness_score]
                })
                df = pd.concat([df, new_entry], ignore_index=True)

                cv.rectangle(frame, (left, top), (right, bottom), 
                           (0, 255, 0) if attentive == 'Yes' else (0, 0, 255), 2)
                cv.putText(frame, name, (left, top - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

        cv.imshow("Video Stream", frame)
        if cv.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if not df.empty:
        average_attention_score = np.mean(attention_scores) if attention_scores else 0.0
        average_liveness_score = np.mean(liveness_scores) if liveness_scores else 0.0
        
        print(f"\nFinal Results:")
        print(f"Average Attention Score: {average_attention_score:.2f}")
        print(f"Average Liveness Score: {average_liveness_score:.2f}")

        is_present = (average_attention_score >= 0.35 and average_liveness_score >= 0.1)
        presence_status = "PRESENT" if is_present else "ABSENT"
        presence_reason = []
        
        if average_attention_score < 0.35:
            presence_reason.append("low attention")
        if average_liveness_score < 0.1:
            presence_reason.append("failed liveness check")
            
        reason_str = " and ".join(presence_reason)
        print(f"\nFinal Status: {presence_status}")
        if not is_present:
            print(f"Reason: {reason_str}")

        avg_entry = pd.DataFrame({
            'Name': ['Average'],
            'Date': [''],
            'Time': [''],
            'Screenshot': [''],
            'Attentive': [presence_status],
            'Attention Score': [average_attention_score],
            'Liveness Score': [average_liveness_score]
        })
        df = pd.concat([df, avg_entry], ignore_index=True)

        df.to_excel('basic_attendance_log.xlsx', index=False)
        print("\nAttendance data saved to 'basic_attendance_log.xlsx'")
    
    cam.release()
    cv.destroyAllWindows() 