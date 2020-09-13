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
                if(name == 'right_eye'):
                    (i,j) = (i,j)
                    break
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = frame[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1_left = cv2.inRange(hsv, (0, 140, 70), (10, 255,255))
            mask2_left = cv2.inRange(hsv, (170, 140, 70), (180, 255,255))
            mask_left = mask1_left + mask2_left
            imask_left = mask_left>0
            red_left = np.zeros_like(roi, np.uint8)
            red_left[imask_left] = roi[imask_left]
            cv2.imshow('right_eye', red_left)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if(name == 'left_eye'):
                    (p,q) = (i,j)
                    break
            (x, y, w, h) = cv2.boundingRect(np.array([shape[p:q]]))
            roi = frame[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1_left = cv2.inRange(hsv, (0, 140, 70), (10, 255,255))
            mask2_left = cv2.inRange(hsv, (170, 140, 70), (180, 255,255))
            mask_left = mask1_left + mask2_left
            imask_left = mask_left>0
            red_left = np.zeros_like(roi, np.uint8)
            red_left[imask_left] = roi[imask_left]
            cv2.imshow('left_eye', red_left)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()