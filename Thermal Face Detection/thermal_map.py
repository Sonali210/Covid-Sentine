import cv2
import numpy as np
import tensorflow as tf
capt = cv2.VideoCapture('a.mp4')
cap=cv2.VideoCapture('b.mp4')
classes = ["unknown","face","background"]
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.gfile.FastGFile('thermal_infer_model.pb','rb') as f:
	graph_def=tf.GraphDef()
	graph_def.ParseFromString(f.read())
with tf.Session() as sess:
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
	while (True):
		_, img = cap.read()
		_, imgt = capt.read()
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
			if (score>0.9):
				x=bbox[1]*cols
				y=bbox[0]*rows
				right=bbox[3]*cols
				bottom=bbox[2]*rows
				color=colors[classId]
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=2)
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				cv2.putText(img, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
				cv2.rectangle(imgt, (int(x), int(y)), (int(right)+25,int(bottom)+25), color, thickness=2)
				cv2.rectangle(imgt, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				cv2.putText(imgt, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
		cv2.imshow('Dashboard',img)
		#cv2.imshow('DashboardT',imgt)
		key=cv2.waitKey(1)
		if (key == 27):
			break
cap.release()
cv2.destroyAllWindows()