import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist
cap = cv2.VideoCapture('p.mp4')
#filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
cv2.namedWindow ('Dashboard', cv2. WINDOW_NORMAL)
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
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.io.gfile.GFile('frozen_inference_graph.pb','rb') as f:
	graph_def=tf.compat.v1.GraphDef()
	graph_def.ParseFromString(f.read())
with tf.compat.v1.Session() as sess:
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
	while (True):
		_, img = cap.read()
		#img =cv2.GaussianBlur(img, (5, 5), 0)
		#img=cv2.filter2D(img,-1,filter)
		rows=img.shape[0]
		cols=img.shape[1]
		inp=cv2.resize(img,(220,220))
		inp=inp[:,:,[2,1,0]]
		out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
				sess.graph.get_tensor_by_name('detection_scores:0'),
                      		sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		sess.graph.get_tensor_by_name('detection_classes:0')],
                     		feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
		#print(out)
		print("*************************************************************************************************")
		num_detections=int(out[0][0])
		print(num_detections)
		mid_pts = []
		for i in range(num_detections):
			#print (i)
			#print ("******************************************")
			classId = int(out[3][0][i])
			score=float(out[1][0][i])
			bbox=[float(v) for v in out[2][0][i]]
			label=classes[classId]
			
			if (score>0.30 and label=='person'):
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
				#print(dist[i][j])
				if((i!=j) & (dist[i][j]<=80.0)):
					cv2.rectangle(img, (int(x1), int(y1)), (int(right1),int(bottom1)), (0,0,255), thickness=2)
					cv2.putText(img,"person",(int(x), int(y+25)),1,2,(255,0,0),2)
					cv2.rectangle(img, (int(x2), int(y2)), (int(right2),int(bottom2)), (0,0,255), thickness=2)
					cv2.putText(img,"person",(int(x), int(y+25)),1,2,(255,0,0),2)
				# else:
				# 	cv2.rectangle(img, (int(x1), int(y1)), (int(right1),int(bottom1)), (0,255,0), thickness=2)
				# 	cv2.rectangle(img, (int(x2), int(y2)), (int(right2),int(bottom2)), (0,255,0), thickness=2)
		cv2.imshow('Dashboard',img)
		key=cv2.waitKey(1)
		if (key == 27):
			break
cap.release()
cv2.destroyAllWindows()
