import cv2
import mediapipe as mp
import csv
import os

# 1. Initialize Skeleton Tracker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 2. Setup Dataset File
csv_file = "asl_dataset.csv"

# Create the file and write the header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create columns: label, x0, y0, z0, x1, y1, z1... up to z20
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# 3. Start Camera
cap = cv2.VideoCapture(0)
current_letter = None
collecting = False
frames_collected = 0

print("--- DATA COLLECTION CONTROLS ---")
print("Press 'a' through 'z' to start recording that letter.")
print("Press the 'SPACEBAR' to pause recording.")
print("Press 'q' to quit and save.")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 4. Extract and Save Data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if collecting and current_letter:
                # Get the wrist coordinates to act as our 0,0 center point
                wrist = hand_landmarks.landmark[0]
                
                # Start building the row of data
                row = [current_letter]
                
                for lm in hand_landmarks.landmark:
                    # Calculate relative coordinates (Normalization)
                    rel_x = lm.x - wrist.x
                    rel_y = lm.y - wrist.y
                    rel_z = lm.z - wrist.z
                    row.extend([rel_x, rel_y, rel_z])
                
                # Append the row to our CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                frames_collected += 1

    # 5. UI Overlay
    status_color = (0, 255, 0) if collecting else (0, 0, 255)
    status_text = f"Recording: {current_letter.upper()}" if collecting and current_letter else "Paused"
    
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    if current_letter:
        cv2.putText(frame, f"Frames saved: {frames_collected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Dataset Generator", frame)

    # 6. Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '): # Spacebar pauses
        collecting = False
    elif 97 <= key <= 122: # If you press a lowercase letter (a-z)
        current_letter = chr(key)
        collecting = True
        frames_collected = 0 # Reset counter for the UI

cap.release()
cv2.destroyAllWindows()
