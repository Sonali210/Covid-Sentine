from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
from imutils import face_utils
import numpy as np
import imutils
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if(name == 'left_eye'):
                (m,n) = (i,j)
                break
        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[m:n]]))
        roi1 = frame[y1:y1 + h1, x1:x1 + w1]
        roi1 = imutils.resize(roi1, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow('left_eye',roi1)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if(name == 'right_eye'):
                (p,q) = (i,j)
                break
        (x2, y2, w2, h2) = cv2.boundingRect(np.array([shape[p:q]]))
        roi2 = frame[y2:y2 + h2, x2:x2 + w2]
        roi2 = imutils.resize(roi2, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow('right_eye',roi2)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()