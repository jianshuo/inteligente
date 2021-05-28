import cv2
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
print(cv2_base_dir)
face_cascade = cv2.CascadeClassifier(os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2_base_dir, 'data', 'haarcascade_eye.xml'))

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
