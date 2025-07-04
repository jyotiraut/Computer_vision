import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm 

cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.handDetector()

while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4]) 

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)