import cv2
import mediapipe as mp
import time 

class FaceDetector:
    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=self.modelSelection,
            min_detection_confidence=self.minDetectionCon
        )

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append((id, bbox, detection.score[0]))

                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',
                                (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bboxs
    
    
    
    

def main():
    cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\faceDetectionBasics\Videos\3.mp4")
    pTime = 0
    detector = FaceDetector(minDetectionCon=0.5)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img)

        # Resize and show FPS
        img = cv2.resize(img, (640, 480))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Face Detection", img)
        cv2.waitKey(1) 

   

if __name__ == "__main__":
    main()

