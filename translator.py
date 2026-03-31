import cv2
import csv
import copy
import time
import itertools
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import tensorflow as tf

print("Loading Pre-Trained TFLite Landmark Model...")

# 1. LOAD THE TENSORFLOW LITE MODEL
interpreter = tf.lite.Interpreter(model_path='keypoint_classifier.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. LOAD THE ASL ALPHABET LABELS
with open('keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

print("✅ Model loaded successfully!")

# 3. SET UP MEDIAPIPE
hand_tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 4. MATH HELPER FUNCTION (Prepares the skeleton for the AI)
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert absolute coordinates to relative coordinates (based on the wrist)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten into a 1D array
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalize the numbers so the AI doesn't care how close or far your hand is
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# 5. START CAMERA
cap = cv2.VideoCapture(0)
current_word = ""
last_letter = ""
start_time = time.time()

print("System Ready! Hold up an ASL sign.")

while True:
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract raw X/Y coordinates of the skeleton
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append([min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)])
                
            # Run the math to format the skeleton for the AI
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Send the 42 numbers into the TFLite Model
            interpreter.set_tensor(input_details[0]['index'], np.array([pre_processed_landmark_list], dtype=np.float32))
            interpreter.invoke()
            tflite_results = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the prediction
            hand_sign_id = np.argmax(np.squeeze(tflite_results))
            predicted_letter = keypoint_classifier_labels[hand_sign_id]
            confidence = np.squeeze(tflite_results)[hand_sign_id]
            
            # Display Prediction
            if confidence > 0.80:
                # Find bounding box for text placement
                x_min = min([coord[0] for coord in landmark_list])
                y_min = min([coord[1] for coord in landmark_list])
                cv2.putText(frame, f"{predicted_letter} ({int(confidence*100)}%)", 
                            (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Word Builder Logic
                if predicted_letter == last_letter:
                    if time.time() - start_time > 1.5:
                        current_word += predicted_letter
                        cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 10) # Flash green
                        start_time = time.time() + 1.0
                else:
                    last_letter = predicted_letter
                    start_time = time.time()

    # UI Overlay
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"TEXT: {current_word}", (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Real-Time Landmark ASL", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()