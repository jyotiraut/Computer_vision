import cv2 
import time 
import mediapipe as mp 


mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, 
                                max_num_faces=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Drawing specs for face landmarks
drawSpecLine = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1)     
drawSpecPoint = mpDraw.DrawingSpec(color=(0, 0, 255), circle_radius=1)

pTime = 0
cap = cv2.VideoCapture(r"C:\Users\acer\OneDrive\Desktop\project CV\faceDetectionBasics\Videos\3.mp4")
# cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
     for faceLms in results.multi_face_landmarks:
        mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpecLine, drawSpecPoint)

        ih, iw, _ = img.shape  
        for id, lm in enumerate(faceLms.landmark):
            x, y = int(lm.x * iw), int(lm.y * ih)
            print(f'Landmark {id}: (x={x}, y={y})')


    img = cv2.resize(img, (640, 480))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)


