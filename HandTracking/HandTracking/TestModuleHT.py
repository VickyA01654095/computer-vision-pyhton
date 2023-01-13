import HandTrackingModule as htm
import cv2
import mediapipe as mp
import time

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # la cámara de mi compu, es 0 porque es la integrada
detector = htm.handDetector()

while True:
    success, img = cap.read()  # necesario para abrir la cámara y mostrarla

    img = detector.findHands(img)
    lmList = detector.findPosition(img)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str("fps: " + str(int(fps))), (10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)

    cv2.imshow("Image", img)  # necesario para abrir la cámara y mostrarla
    cv2.waitKey(1)  # necesario para abrir la cámara y mostrarla