import cv2 
import time 
import poseModule as pm 
pTime = 0
cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\video2.mp4")
detector = pm.poseDetector()

while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.getPosition(img,draw=False)
        print(lmList[14])
        cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
          