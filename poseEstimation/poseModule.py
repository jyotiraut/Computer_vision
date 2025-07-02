import cv2 
import mediapipe as mp
import time 


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
        lmList = []
        for id , lm in enumerate (self.results.pose_landmarks.landmark):
            h,w,c = img.shape
        
            cx,cy = int(lm.x*w),int(lm.y*h)
            lmList.append([id,cx,cy])
            if draw:

              cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
            return lmList

def main():
    pTime = 0
    cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\poseEstimation\poseVideos\video2.mp4")
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
          


if __name__ == "__main__":
    main()
