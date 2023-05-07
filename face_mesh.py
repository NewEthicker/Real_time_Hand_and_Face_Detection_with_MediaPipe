import cv2
import mediapipe as mp

# MediaPipe Hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, min_tracking_confidence=0.99, min_detection_confidence=0.3)
mpDraw = mp.solutions.drawing_utils

# MediaPipe Face Mesh detection
facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=False, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

screen_width, screen_height = 1000, 600
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", screen_width, screen_height)

# Video capture from URL
url = "https://10.43.160.126:8080/video"   # change it from your ip of webcame, dont have app: https://play.google.com/store/apps/details?id=com.pas.webcam
cap = cv2.VideoCapture(url)

while True:
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            draw.draw_landmarks(frame, i, facemesh.FACE_CONNECTIONS,
                                landmark_drawing_spec=draw.DrawingSpec(color=(10, 255, 25), circle_radius=1))
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            thumb_x = handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x
            thumb_y = handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y
            index_x = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
            index_y = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y

            print("Thumb Landmark: (x={}, y={})".format(thumb_x, thumb_y))
            print("Index Finger Landmark: (x={}, y={})".format(index_x, index_y))

            if thumb_y < 0.40:
                print("GOOD")
                h, w, c = frame.shape
                x, y, _ = handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[0].z
                x, y = int(x * w), int(y * h)
                cv2.putText(frame, "GOOD", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (5, 5, 255), 7)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 0, (0, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", frame)
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

