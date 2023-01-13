import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode = False, enable_segmentation = False, smooth = True,smooth_segmentation=True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.enable_segmentation = enable_segmentation
        self.smooth = smooth
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,1, self.smooth, self.enable_segmentation,self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw= True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if draw:
            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):

        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int (lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 0, 0), cv2.FILLED)

        return lmList
    

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        lmList = detector.getPosition(img)
        img = detector.findPose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str("fps: " + str(int(fps))), (10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()