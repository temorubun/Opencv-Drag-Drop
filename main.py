import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0) 
cap.set(3, 1280) 
cap.set(4, 720)  

detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

class DragShape():
    def __init__(self, posCenter, size=[200, 200], shape='rectangle'):
        self.posCenter = posCenter
        self.size = size
        self.shape = shape
        self.isDragging = False

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor
            self.isDragging = True
        else:
            self.isDragging = False

    def toggle_shape(self):
        if self.shape == 'rectangle':
            self.shape = 'circle'
        else:
            self.shape = 'rectangle'

shapeList = []
for x in range(5):
    shapeList.append(DragShape([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img) 
    if hands:
        lmList = hands[0]['lmList'] 
        index_finger_tip = lmList[8][:2]  
        middle_finger_tip = lmList[12][:2] 
        l, _, _ = detector.findDistance(index_finger_tip, middle_finger_tip, img)  
        if l < 30:
            cursor = index_finger_tip  
            for shape in shapeList:
                shape.update(cursor)

        thumb_tip = lmList[4][:2]
        thumb_index_dist, _, _ = detector.findDistance(thumb_tip, index_finger_tip, img)
        if thumb_index_dist < 40:
            for shape in shapeList:
                if shape.isDragging:
                    shape.toggle_shape()

    imgNew = np.zeros_like(img, np.uint8)
    for shape in shapeList:
        cx, cy = shape.posCenter
        w, h = shape.size
        if shape.shape == 'rectangle':
            cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), colorR if not shape.isDragging else (0, 255, 0), cv2.FILLED)
            cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
        elif shape.shape == 'circle':
            cv2.circle(imgNew, (cx, cy), w // 2, colorR if not shape.isDragging else (0, 255, 0), cv2.FILLED)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
