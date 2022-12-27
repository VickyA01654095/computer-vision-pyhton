import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0) # la cámara de mi compu, es 0 porque es la integrada

mpHands = mp.solutions.hands
hands = mpHands.Hands(False) # tracking and detection params, check doc
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read() # necesario para abrir la cámara y mostrarla
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                '''
                # highlight certain point in hand with id
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                '''

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str("fps: " + str(int(fps))), (10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)

    cv2.imshow("Image", img) # necesario para abrir la cámara y mostrarla
    cv2.waitKey(1) # necesario para abrir la cámara y mostrarla