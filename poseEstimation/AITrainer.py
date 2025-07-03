import cv2 
import numpy as np 
import mediapipe as mp 
import poseModule as pm 



cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\curl.mp4")
detector = pm.poseDetector()
count = 0 
dir = 0

while True :
    success , img = cap.read()
    img = cv2.resize(img, (640, 480))
    # img = cv2.imread(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\1.jpg")
    img = detector.findPose(img,False)
    lmList = detector.getPosition(img,False)
    # print(lmLIst)
    if len(lmList) !=0:
        # detector.findAngle(img,12,14,16)

        angle = detector.findAngle(img,11,13,19)
        per = np.interp(angle,(210,310),(0,100))
        print(angle,per)

        #check for the dumbell curls 
        if per==100:
            if dir == 0:
                count +=0.5
                dir = 1
        if per ==0:
            if dir ==1:
                count+=0.5
                dir = 0
                print(count)
                cv2.putText(img,str(int(count)),(50,100),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),5)

        


    
    cv2.imshow("Image",img)
    cv2.waitKey(1)



