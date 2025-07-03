import cv2 
import time 
import mediapipe as mp 

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


pTime = 0
cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\faceDetectionBasics\Videos\3.mp4")

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

  
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    img = cv2.resize(img, (640, 480))

  
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(1) 
       


