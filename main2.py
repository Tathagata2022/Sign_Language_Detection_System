import cv2
import mediapipe as mp
import pickle
import warnings

# Suppress standard Scikit-Learn terminal warnings to keep things clean
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load your newly trained AI model
print("Loading AI Brain...")
try:
    with open('asl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ AI Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: 'asl_model.pkl' not found.")
    exit()

# 2. Initialize MediaPipe Skeleton Tracker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 3. Start Camera
cap = cv2.VideoCapture(0)
print("Ready! Show your ASL signs to the camera. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success: break
    
    # Flip frame like a mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process skeleton
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- THE AI PIPELINE ---
            
            # 1. Find the wrist (our 0,0 anchor point)
            wrist = hand_landmarks.landmark[0]
            
            # 2. Extract the 63 relative coordinates (just like we did in training!)
            row = []
            for lm in hand_landmarks.landmark:
                rel_x = lm.x - wrist.x
                rel_y = lm.y - wrist.y
                rel_z = lm.z - wrist.z
                row.extend([rel_x, rel_y, rel_z])
            
            # 3. Ask the AI to predict the letter based on those 63 numbers
            prediction = model.predict([row])[0]
            
            # --- UI OVERLAY ---
            # Draw a sleek black box and display the predicted letter in bright green
            cv2.rectangle(frame, (10, 10), (150, 100), (0, 0, 0), -1)
            cv2.putText(frame, prediction.upper(), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    cv2.imshow("Real-Time AI ASL Translator", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# collect_data.py
# train_model.py
# main2.py