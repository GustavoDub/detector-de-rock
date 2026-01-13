import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ve os dedos abertos
def dedos_abertos(hand_landmarks):
    dedos = []

    #  referencia polegar
    dedos.append(
        hand_landmarks.landmark[4].x <
        hand_landmarks.landmark[3].x
    )

    #  outros dedo
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        dedos.append(
            hand_landmarks.landmark[tip].y <
            hand_landmarks.landmark[pip].y
        )

    return dedos.count(True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    rock_pose = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            if dedos_abertos(hand_landmarks) >= 4:
                rock_pose = True

    if rock_pose:
        cv2.putText(
            frame,
            "TU E DO ROCK",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
            cv2.LINE_AA
        )

    cv2.imshow("Detector de Rock 🤘", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
