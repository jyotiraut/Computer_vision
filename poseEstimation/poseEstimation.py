import cv2 
import mediapipe as mp
import time 

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose 
pose = mpPose.Pose()

pTime = 0

cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\video4.mp4")

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    img = cv2.resize(img, (640, 360))
  

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id , lm in enumerate (results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx,cy = int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 

    
    cv2.putText(img,str(int(fps)),(70,80),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
   
    cv2.imshow("Image",img)
    cv2.waitKey(1)