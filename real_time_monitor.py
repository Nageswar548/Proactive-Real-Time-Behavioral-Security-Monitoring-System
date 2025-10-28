# real_time_monitor.py (FINAL ACTION DETECTION SCRIPT)

from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- 1. Load Trained Model and Encoder ---
MODEL_PATH = os.path.join('security_action_model.pkl')
ENCODER_PATH = os.path.join('action_encoder.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Trained model and encoder loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run train_model.py first.")
    exit()

# --- 2. Setup Models and Libraries (Same as before) ---
model = YOLO('yolov8n.pt') 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
num_coords = 33 

# Function to extract landmark coordinates (33 joints * 3 coords = 99 features)
def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    return np.zeros(num_coords*3)

# --- 3. Start Webcam Stream ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- STARTING REAL-TIME SECURITY MONITOR ---")
print("Press 'q' to quit.")

# --- 4. Process Frames and Predict Action ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4.1. Get person bounding box using YOLO
    # We use persist=True to maintain person ID across frames (for tracking)
    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
    
    # Default display text
    predicted_action = "DETECTING..."
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        if len(boxes) > 0:
            x_min, y_min, x_max, y_max = boxes[0] 
            person_roi = frame[y_min:y_max, x_min:x_max]
            
            # 4.2. Process ROI for Pose Estimation
            img_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(img_rgb)
            
            # 4.3. Get Keypoints and Predict Action
            if pose_results.pose_landmarks:
                # 1. Extract features
                keypoints = extract_keypoints(pose_results)
                
                # 2. Reshape for the model (needs 1 row, 99 columns)
                X_input = keypoints[np.newaxis, :] 
                
                # 3. Predict the action
                action_number = classifier.predict(X_input)[0]
                
                # 4. Convert number back to action name (text)
                predicted_action = label_encoder.inverse_transform([action_number])[0]
                
                # Draw landmarks on the ROI
                mp.solutions.drawing_utils.draw_landmarks(
                    person_roi, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS)

            # Draw the predicted action text over the bounding box
            cv2.putText(frame, predicted_action, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
    # Display the result (YOLO handles drawing the bounding box)
    annotated_frame = results[0].plot()
    cv2.imshow("Real-Time Security Action Monitor (Step 6)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")