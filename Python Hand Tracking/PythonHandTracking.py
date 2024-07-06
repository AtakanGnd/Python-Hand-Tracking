import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        fingers = []

        if landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)
        else:
            fingers.append(0)

        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_dips = [
            mp_hands.HandLandmark.INDEX_FINGER_DIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp_hands.HandLandmark.RING_FINGER_DIP,
            mp_hands.HandLandmark.PINKY_DIP
        ]

        for tip, dip in zip(finger_tips, finger_dips):
            if landmarks[tip].y < landmarks[dip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)
    return 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamıyor.")
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(results.multi_hand_landmarks)

    cv2.putText(image, f'Sayi: {finger_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)

    cv2.imshow('Parmak Sayisi Algilama', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
