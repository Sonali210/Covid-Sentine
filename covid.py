from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter.font import Font
from PIL import ImageTk,Image
from tkinter import simpledialog
from tkinter import messagebox
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import tensorflow as tf
from scipy.spatial import distance
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

root = tk.Tk()
root.title("Covid-19 Solution Compliance")
root.geometry("1680x1080")
#root.attributes('-fullscreen',True)
root.configure(bg='#E6E6FA')
#To add bg image
C = Canvas(root, bg="blue", height=250, width=300)
filename = ImageTk.PhotoImage(Image.open("img.png"))
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.place(relx=0.5, rely=0.5, anchor=CENTER)

#To add logo if any
#img2 = ImageTk.PhotoImage(Image.open('logo.jpg'))

def eyes_detection():
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
def redness_detection():
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
def mask_detection():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow ('Dashboard', cv2. WINDOW_NORMAL)
    cv2.setWindowProperty ('Dashboard', cv2. WND_PROP_FULLSCREEN, cv2. WINDOW_FULLSCREEN)
    classes = ["unknown","mask","no_mask"]
    # colors = np.random.uniform(0,255,size=(len(classes),3))
    with tf.gfile.FastGFile('xyz.pb','rb') as f:
	    graph_def=tf.GraphDef()
	    graph_def.ParseFromString(f.read())
    with tf.Session() as sess:
	    sess.graph.as_default()
	    tf.import_graph_def(graph_def, name='')
	    while (True):
		    _, img = cap.read()
		    rows=img.shape[0]
		    cols=img.shape[1]
		    inp=cv2.resize(img,(220,220))
		    inp=inp[:,:,[2,1,0]]
		    out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
				    sess.graph.get_tensor_by_name('detection_scores:0'),
                      		    sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		    sess.graph.get_tensor_by_name('detection_classes:0')],
                     		    feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
		    num_detections=int(out[0][0])
	
		    for i in range(num_detections):
			    classId = int(out[3][0][i])
			    score=float(out[1][0][i])
			    bbox=[float(v) for v in out[2][0][i]]
			    label=classes[classId]
			    mask = 0 
			    no_mask = 0
			    if(classId == 2):
				    no_mask = no_mask + 1
				    color = (0,0,255)
			    else:
				    color = (0,255,0)
				    mask = mask + 1
			
			    if (score>0.6):
				    x=bbox[1]*cols
				    y=bbox[0]*rows
				    right=bbox[3]*cols
				    bottom=bbox[2]*rows
				
				
				    cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=1)
				    cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				    cv2.putText(img, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
				    cv2.putText(img, "MASK : "+str(int(mask)), (50, 440),1 , 1, (255, 0, 0), 2, cv2.LINE_4)
				    cv2.putText(img, "NO_MASK : "+str(int(no_mask)), (50, 460),1 , 1, (255, 0, 0), 2, cv2.LINE_4)
		    cv2.imshow('Dashboard',img)
		    if cv2.waitKey(1) == ord('q'):
			    break
    cap.release()
    cv2.destroyAllWindows()
def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist
def social_distancing():
    cap = cv2.VideoCapture(0)
    classes = ["background","person",   "bicycle",   "car",   "motorcycle",   "airplane",
        "bus",   "train",   "truck",   "boat",   "traffic light",   "fire hydrant","unknown", "stop_sign",
        "parking meter",   "bench",   "bird",   "cat",   "dog",   "horse",   "sheep",   "cow","elephant", "bear",
        "zebra", "giraffe","unknown", "backpack", "umbrella","unknown","unknown",   "handbag",   "tie",   "suitcase",   "frisbee","skis", "snowboard",
        "sports ball",   "kite",   "baseball bat",   "baseball glove",   "skateboard",   "surfboard", "tennis racket",
        "bottle","unknown","wine glass",   "cup",  "fork",   "knife",   "spoon",   "bowl",   "banana",   "apple",   "sandwich",   "orange",
        "broccoli",   "carrot",   "hot dog",   "pizza",   "donot",   "cake",   "chair",   "couch",   "potted plant",   "bed",
        "unknown","dining table","unknown","unknown",  "toilet","unknown", "tv",   "laptop",   "mouse",   "remote",   "keyboard",   "cell phone",   "microwave",
        "oven",   "toaster",   "sink",   "refrigerator","unknown",   "book",   "clock",   "vase",   "scissors",   "teddy bear",   "hair dryer",
        "toothbrush"]
# colors = np.random.uniform(0,255,size=(len(classes),3))
    with tf.io.gfile.GFile('frozen_inference_graph.pb','rb') as f:
	    graph_def=tf.compat.v1.GraphDef()
	    graph_def.ParseFromString(f.read())
    with tf.compat.v1.Session() as sess:
	    sess.graph.as_default()
	    tf.import_graph_def(graph_def, name='')
	    while (True):
		    _, img = cap.read()
		    rows=img.shape[0]
		    cols=img.shape[1]
		    inp=cv2.resize(img,(220,220))
		    inp=inp[:,:,[2,1,0]]
		    out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
				    sess.graph.get_tensor_by_name('detection_scores:0'),
                      		sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		sess.graph.get_tensor_by_name('detection_classes:0')],
                     		feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
		    num_detections=int(out[0][0])
		    mid_pts = []
		    for i in range(num_detections):
			    classId = int(out[3][0][i])
			    score=float(out[1][0][i])
			    bbox=[float(v) for v in out[2][0][i]]
			    label=classes[classId]
			
			    if (score>0.60 and label=='person'):
				    x=bbox[1]*cols
				    y=bbox[0]*rows
				    right=bbox[3]*cols
				    bottom=bbox[2]*rows
				    x_co = (x+right) / 2
				    y_co = (y+bottom) / 2
				    mid_pts.append([x_co,y_co])
								
				# color=colors[classId]
				    cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), (0,255,0), thickness=2)
				# print(x,y,right,bottom)
				    cv2.putText(img,"person",(int(x), int(y+25)),1,2,(255,255,255),2)
				# print(out[2][0])
		    dist = compute_distance(mid_pts,len(mid_pts))
		    for i in range(len(mid_pts)):
			    for j in range(i,len(mid_pts)):
				    x1 = out[2][0][i][1]*cols
				    y1= out[2][0][i][0]*rows
				    right1=out[2][0][i][3]*cols
				    bottom1=out[2][0][i][2]*rows
				    x2 = out[2][0][j][1]*cols
				    y2= out[2][0][j][0]*rows
				    right2=out[2][0][j][3]*cols
				    bottom2=out[2][0][j][2]*rows
				    # print(dist[i][j])
				    if((i!=j) & (dist[i][j]<=60.0)):
					    cv2.rectangle(img, (int(x1), int(y1)), (int(right1),int(bottom1)), (0,0,255), thickness=2)
					    cv2.putText(img,"person",(int(x), int(y+25)),1,2,(255,0,0),2)
					    cv2.rectangle(img, (int(x2), int(y2)), (int(right2),int(bottom2)), (0,0,255), thickness=2)
					    cv2.putText(img,"person",(int(x), int(y+25)),1,2,(255,0,0),2)
				# else:
				# 	cv2.rectangle(img, (int(x1), int(y1)), (int(right1),int(bottom1)), (0,255,0), thickness=2)
				# 	cv2.rectangle(img, (int(x2), int(y2)), (int(right2),int(bottom2)), (0,255,0), thickness=2)
		    cv2.imshow('Dashboard',img)
		    if cv2.waitKey(1) == ord('q'):
			    break
    cap.release()
    cv2.destroyAllWindows()
# def recognition():
# 	j = simpledialog.askinteger(title="Day", prompt="Input the day:")
# 	sheet = pd.read_csv('data.csv')
# 	encodings = 'encodings.pickle'
# 	print("[INFO] loading encodings...")
# 	data = pickle.loads(open(encodings, "rb").read())
# 	print("[INFO] starting video stream...")
# 	vs = VideoStream(src=0).start()
# 	writer = None
# 	while True:
# 		frame = vs.read()
# 		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		rgb = imutils.resize(frame, width=750)
# 		r = frame.shape[1] / float(rgb.shape[1])
# 		boxes = face_recognition.face_locations(rgb,model="hog")
# 		encodings = face_recognition.face_encodings(rgb, boxes)
# 		names = []
# 		for encoding in encodings:
# 			matches = face_recognition.compare_faces(data["encodings"], encoding)
# 			name = "Unknown"
# 			if True in matches:
# 				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
# 				counts = {}
# 				for i in matchedIdxs:
# 					name = data["names"][i]
# 					counts[name] = counts.get(name, 0) + 1
# 				name = max(counts, key=counts.get)
# 			names.append(name)
# 		# loop over the recognized faces
# 		for ((top, right, bottom, left), name) in zip(boxes, names):
# 		#rescale the face coordinates
# 		    top = int(top * r)
# 		    right = int(right * r)
# 		    bottom = int(bottom * r)
# 		    left = int(left * r)
# 		    # draw the predicted face name on the image
# 		    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
# 		    y = top - 15 if top - 15 > 15 else top + 15
# 		    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
# 		    pname = str(name)
# 		    for i in range(0,22):
# 		    	if(sheet.iloc[i,0] == pname):
# 		    		sheet.iloc[i,j] = 1
# 		# if the video writer is None *AND* we are supposed to write
# 		# the output video to disk initialize the writer
# 		if writer is None is not None:
# 			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# 			writer = cv2.VideoWriter(args["output"], fourcc, 20,(frame.shape[1], frame.shape[0]), True)
# 		# if the writer is not None, write the frame with recognized faces t odisk
# 		if writer is not None:
# 			writer.write(frame)
# 		# check to see if we are supposed to display the output frame to the screen
# 		if (1>0):
# 			cv2.imshow("Frame", frame)
# 			key = cv2.waitKey(1) & 0xFF
# 			# if the `q` key was pressed, break from the loop
# 			if key == ord("q"):
# 				break
# 	# do a bit of cleanup
# 	cv2.destroyAllWindows()
# 	vs.stop()
# 	# check to see if the video writer point needs to be released
# 	if writer is not None:
# 		writer.release()
# 	print(sheet)
#fn stop requires amendment
def stop():
	cv2.destroyAllWindows()

'''def exit():
	answer=messagebox.askquestion("exit","Do you sure to exit")
	if answer == 'yes':
		root.quit()	'''

my_font=Font(family="Helvetica",size=30,weight="bold",slant="italic")
label=Label(text="AI Based Covid-19 Solution Compliance",font=my_font,foreground="midnightblue",background="gold2").place(relx=0.5, rely=0.1, anchor=CENTER)

#canvas for image on leftframe
'''canvas=Canvas(leftframe,width=540,height=470)
canvas.pack()
photo=PhotoImage(file='erp.png')
canvas.create_image(0,30,image=photo,anchor=NW)
'''
#rightframe=Frame(bottomframe,padx=10)
Bt1=Button(text="Detect Eyes" , width=35, bd=1 ,pady=10 , activebackground='yellow3', relief=RAISED, bg='yellow2', fg='midnightblue', command=eyes_detection)
Bt1.place(relx=0.3, rely=0.3, anchor=CENTER)
Bt2=Button(text="Detect Redness" , width=35, bd=1 ,pady=10 , activebackground='yellow3', relief=RAISED, bg='yellow2', fg='midnightblue', command=redness_detection)
Bt2.place(relx=0.3, rely=0.4, anchor=CENTER)
Bt3=Button(text="Detect Face Mask" , width=35, bd=1 ,pady=10 , activebackground='yellow3', relief=RAISED, bg='yellow2', fg='midnightblue', command=mask_detection)
Bt3.place(relx=0.3, rely=0.5, anchor=CENTER)
Bt4=Button(text="Monitor Social Distancing" , width=35, bd=1 ,pady=10 , activebackground='yellow3', relief=RAISED, bg='yellow2', fg='midnightblue', command=social_distancing)
Bt4.place(relx=0.3, rely=0.6, anchor=CENTER)
#Bt5=Button(text="Stop",width=35, bd=1 ,pady=10, activebackground='yellow3', relief=RAISED, bg='yellow2', fg='midnightblue', command=stop)
#Bt5.place(relx=0.3, rely=0.6, anchor=CENTER)
# Bt3=Button(,text="Attendance File",width=35, bd=1 ,pady=10 , activebackground="green", relief=RAISED)
# Bt3.place(relx=0.8, rely=0.92, anchor=CENTER)
#Bt6=Button(text="EXIT",width=35, bd=1 ,pady=10 , activebackground="green", command=exit, relief=RAISED)
#Bt6.place(relx=0.5, rely=0.7, anchor=CENTER)	

#leftframe.pack(side=LEFT)
#.pack(side=RIGHT)
#bottomframe.pack()
root.mainloop()
