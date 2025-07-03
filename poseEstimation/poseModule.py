import cv2 
import mediapipe as mp
import time 
import math


class poseDetector():
    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode 
        self.smooth = smooth 
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose 
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        img = cv2.resize(img, (640, 360))

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def getPosition(self,img,draw=True):
        self.lmList = []
        for id , lm in enumerate (self.results.pose_landmarks.landmark):
            h,w,c = img.shape
        
            cx,cy = int(lm.x*w),int(lm.y*h)
            self.lmList.append([id,cx,cy])
            if draw:

              cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
        return self.lmList
    
    def findAngle(self,img,p1,p2,p3,draw = True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]


        #calculate the angle 
        angle = math.degrees(math.atan2(y3-y3,x3-x2)-math.atan2(y1-y2,x1-x2))
        # print(angle)
        if angle<0:
            angle+=360

        if draw:

              cv2.circle(img,(x1,y1),5,(255,0,0),cv2.FILLED)
              cv2.circle(img,(x1,y1),10,(255,0,0),2)
              cv2.circle(img,(x2,y2),5,(255,0,0),cv2.FILLED)
              cv2.circle(img,(x2,y2),10,(255,0,0),2)
              cv2.circle(img,(x3,y3),5,(255,0,0),cv2.FILLED)
              cv2.circle(img,(x3,y3),10,(255,0,0),2)
              cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
              cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 3)
              cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)
        return angle 

def main():
    pTime = 0
    cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\video2.mp4")
    detector = poseDetector()

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
          


if __name__ == "__main__":
    main()
