import cv2 
import mediapipe as mp
import time 

pTime = 0

cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\video2.mp4")

while True:
    success,img = cap.read()
    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image",img)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 

    cv2.putText(img,str(int(fps)),(70,80),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    cv2.waitKey(1)