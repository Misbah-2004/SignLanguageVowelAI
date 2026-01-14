import cv2
import mediapipe as mp
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Sign Language Vowel Learning (Live Demo)")

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def predict_vowel(hand):
    wrist = hand.landmark[0]
    palm = hand.landmark[9]   # middle finger base (stable palm reference)

    thumb = hand.landmark[4]
    index = hand.landmark[8]
    middle = hand.landmark[12]
    ring = hand.landmark[16]
    pinky = hand.landmark[20]

    # Distances from wrist
    d_index = distance(index, wrist)
    d_middle = distance(middle, wrist)
    d_ring = distance(ring, wrist)
    d_pinky = distance(pinky, wrist)

    # Palm size reference
    palm_size = distance(palm, wrist)

    index_open = d_index > palm_size * 1.2
    middle_open = d_middle > palm_size * 1.2
    ring_open = d_ring > palm_size * 1.2
    pinky_open = d_pinky > palm_size * 1.2

    # Distances to thumb (for O)
    t_index = distance(index, thumb)
    t_middle = distance(middle, thumb)
    t_ring = distance(ring, thumb)
    t_pinky = distance(pinky, thumb)

    thumb_avg = (t_index + t_middle + t_ring + t_pinky) / 4

    # ---- ASL VOWEL RULES ----

    # ✅ A → fist (ALL fingers close to palm)
    if (
        d_index < palm_size * 1.1 and
        d_middle < palm_size * 1.1 and
        d_ring < palm_size * 1.1 and
        d_pinky < palm_size * 1.1
    ):
        return "A"

    # I → pinky only
    if pinky_open and not index_open and not middle_open and not ring_open:
        return "I"

    # U → index + middle
    if index_open and middle_open and not ring_open and not pinky_open:
        return "U"

    # E → index + middle + ring
    if index_open and middle_open and ring_open and not pinky_open:
        return "E"

    # O → all fingers bent toward thumb
    if thumb_avg < palm_size * 0.9:
        return "O"

    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "Show vowel sign"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            vowel = predict_vowel(hand)
            text = f"Vowel: {vowel}"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Vowel Learning (Live Demo)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('e'), ord('E')]:
        break

cap.release()
cv2.destroyAllWindows()
