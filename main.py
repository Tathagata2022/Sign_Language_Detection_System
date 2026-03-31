import cv2
import mediapipe as mp
import math

# 1. Initialize Skeleton Tracker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Finger Landmark IDs (Tip and Joint below it)
FINGER_TIPS = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
FINGER_JOINTS = [6, 10, 14, 18]
THUMB_TIP = 4
THUMB_JOINT = 2

def get_distance(landmark_1, landmark_2):
    """Calculates the mathematical distance between two points."""
    return math.hypot(landmark_1.x - landmark_2.x, landmark_1.y - landmark_2.y)

def get_finger_states(hand_landmarks):
    """
    Analyzes the skeleton to see which fingers are UP (1) or DOWN (0).
    Returns a list of 5 numbers: [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    
    # 1. Check Thumb (Based on X coordinate for sideways movement)
    if hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_JOINT].x:
        fingers.append(1) # Thumb is OUT
    else:
        fingers.append(0) # Thumb is IN
        
    # 2. Check Other 4 Fingers (Based on Y coordinate for up/down movement)
    for tip, joint in zip(FINGER_TIPS, FINGER_JOINTS):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            fingers.append(1) # Finger is UP
        else:
            fingers.append(0) # Finger is DOWN
            
    return fingers

def recognize_letter(finger_states, hand_landmarks):
    """
    Matches the pattern of open/closed fingers and joint distances to an ASL letter.
    """
    # Grab the specific fingertips we need to measure
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    
    # --- Distance-Based Letters ---
    
    # 1. Check for 'C' (Thumb and Index are curved and relatively close)
    pinch_distance = get_distance(thumb_tip, index_tip)
    if finger_states == [1, 0, 0, 0, 0] or finger_states == [0, 0, 0, 0, 0]:
        # If the gap is between these specific thresholds, it's a C
        if 0.05 < pinch_distance < 0.20: 
            return "C"

    # 2. Distinguish between 'U' and 'V' (Both have Index and Middle up)
    if finger_states == [0, 1, 1, 0, 0] or finger_states == [1, 1, 1, 0, 0]:
        uv_distance = get_distance(index_tip, middle_tip)
        if uv_distance > 0.04: # Fingers are spread apart
            return "V"
        else: # Fingers are pressed together
            return "U"
            
    # --- Standard State-Based Letters ---
    if finger_states == [1, 0, 0, 0, 0]: return "A"
    if finger_states == [0, 1, 1, 1, 1]: return "B"
    if finger_states == [0, 1, 0, 0, 0]: return "D"
    if finger_states == [0, 0, 0, 0, 0]: return "E"
    if finger_states == [1, 0, 1, 1, 1]: return "F"
    if finger_states == [0, 0, 0, 0, 1]: return "I"
    if finger_states == [1, 1, 0, 0, 0]: return "L"
    if finger_states == [0, 1, 1, 1, 0]: return "W"
    if finger_states == [1, 0, 0, 0, 1]: return "Y"
    
    return "..." 

# 2. Start Camera
cap = cv2.VideoCapture(0)
print("Skeleton Tracking Active. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success: break
    
    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the hand skeleton
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton lines
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 1. Get the up/down state of the fingers
            states = get_finger_states(hand_landmarks)
            
            # 2. Pass both the states and the raw skeleton to the recognition function
            letter = recognize_letter(states, hand_landmarks)
            
            # 3. Draw the black box and green text on the screen
            cv2.rectangle(frame, (10, 10), (150, 100), (0, 0, 0), -1)
            cv2.putText(frame, letter, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            # Optional: Print raw data to terminal to help you calibrate distances
            # print(f"States: {states} | Pinch Dist: {get_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8]):.3f}")

    cv2.imshow("Rule-Based ASL Translator", frame)
    
    # Press 'q' to exit the camera window safely
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()