import cv2
import numpy as np
import tensorflow as tf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(0)
classes = ["background","number plate"]
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.io.gfile.GFile('num_plate.pb','rb') as f:
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
		for i in range(num_detections):
			classId = int(out[3][0][i])
			score=float(out[1][0][i])
			bbox=[float(v) for v in out[2][0][i]]
			label=classes[classId]
			if (score>0.3):
				x=bbox[1]*cols
				y=bbox[0]*rows
				right=bbox[3]*cols
				bottom=bbox[2]*rows
				color=colors[classId]
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=1)
				#cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				#cv2.putText(img, str(label),(int(x), int(y)),1,2,(255,255,255),2)
				crop = img[int(y):int(bottom), int(x):int(right)]
				gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
				Cropped = cv2.resize(gray,(300,100))
				#edged = cv2.Canny(Cropped, 30, 150)
				#thresh = cv2.threshold(gray, 225, 200, cv2.THRESH_BINARY_INV)[1]
				cv2.imshow('croped', Cropped)
				text = pytesseract.image_to_string(Cropped, config='--psm 11')
				print("Detected license plate Number is:",text) 
		cv2.imshow('Dashboard',img)
		key=cv2.waitKey(1)
		if (key == 27):
			break
cap.release()
cv2.destroyAllWindows()