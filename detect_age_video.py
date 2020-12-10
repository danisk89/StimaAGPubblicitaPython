# python detect_age_video.py --face face_detector --age age_detector
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json

def detect_and_predict_age(frame, faceNet, ageNet, gender_net, minConf=0.5):
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	results = []

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > minConf:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]

			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]
			
			gender_net.setInput(faceBlob)
			gender_preds = gender_net.forward()
			gender_list = ['Male', 'Female']
			gender = gender_list[gender_preds[0].argmax()]
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence),
				"gender": (gender)
			}
			results.append(d)

	return results
	
def get_fromA(data):
	data = data.replace('(','')
	data = data.replace(')','')
	data0 = int(data.split("-")[1])
	data1 = int(data.split("-")[0])
	return (data0+data1)/2
	
def change_pic(avg_age, num_pic, path, gender):
	if(num_pic==-1):
		#scelgo una foto a caso
		num_pic = 0
	else:
		#foto successiva
		num_pic = num_pic +1
	#print(num_pic)
	ind = 0
	for i in PATH:
		if int(path[ind]["min"])<=avg_age and avg_age<=int(path[ind]["max"]) and (path[ind]["gender"]==gender or path[ind]["gender"]=="U"):
			break
		ind = ind + 1
	if(ind<=len(path)):
		tot_arr = os.listdir(path[ind]["path"])
		#print(len(tot_arr))
		if len(tot_arr) >= (num_pic+1) :
			path = path[ind]["path"]+"/"+tot_arr[num_pic]
		else:
			num_pic=0
			path = path[ind]["path"]+"/"+tot_arr[num_pic]
	else:
		path=path['default']
		num_pic=0
	return num_pic, path


with open('/home/pi/StimaAGPubblicitaPython/custom.json') as f:
  JSON = json.load(f)	
THRS = JSON["THRS"]
THRS_FRAME_RATE = JSON["THRS_FRAME_RATE"]
SMALL_IMAGE = JSON["SMALL_IMAGE"]
PATH = JSON["PATH"]
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

gender_net = cv2.dnn.readNetFromCaffe(
		'/home/pi/StimaAGPubblicitaPython/data/deploy_gender.prototxt', 
		'/home/pi/StimaAGPubblicitaPython/data/gender_net.caffemodel')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture('/home/pi/StimaAGPubblicitaPython/videoPro.mp4')
time.sleep(2.0)
avg_age = 0
old_avg_age = 0
old_pic_num = 0
old_gender="M"
index = 0
init_frame = cv2.imread(JSON['default'])
while True:
	frame = vs.read()
	#ret, frame = vs.read()
	frame = imutils.resize(frame, width=400)

	results = detect_and_predict_age(frame, faceNet, ageNet, gender_net, 
		minConf=args["confidence"])

	indexFace=1
	avg_Frame = 0;
	avg_gender_f =0
	avg_gender_m =0
	now_gender = "M"
	for r in results:
		text = "{}: {:.2f}%, {}".format(r["age"][0], r["age"][1] * 100,r["gender"][0])
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		avg_Frame = (avg_Frame+get_fromA(r["age"][0]))/indexFace
		indexFace=indexFace+1
		if r["gender"][0]=="M" :
			avg_gender_m=avg_gender_m+1
		else:
			avg_gender_f=avg_gender_f+1
		#avg_age = (avg_age+get_fromA(r["age"][0]))/index
		#print(str(avg_age))
	if avg_gender_f>=avg_gender_m :
		now_gender = "F"
	avg_age = (avg_age +avg_Frame)/2
	#print(len(results))
	if(len(results)==0):
		avg_age=0
		old_pic_num=0
		path_real=JSON['default']
		#print(avg_age)
	else: 
		if(index%THRS_FRAME_RATE==0):
			path_real = ""
			if((old_avg_age+THRS)>=avg_age and (old_avg_age-THRS)<=avg_age and old_gender==now_gender):
				old_pic_num, path_real = change_pic(avg_age, old_pic_num, PATH, now_gender)
			else:
				old_pic_num, path_real = change_pic(avg_age, -1, PATH, now_gender)
			old_avg_age = avg_age
			old_gender = now_gender
			#print(path_real)
			init_frame = cv2.imread(path_real)
			#print(old_pic_num)
			#print(old_avg_age)
	index = index+1
	#cv2.imshow("Frame", frame)
	
	cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow("window", init_frame)
	if(SMALL_IMAGE==1):
		cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
#vs.stop()
