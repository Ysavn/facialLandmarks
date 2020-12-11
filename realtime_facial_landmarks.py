#code tested on Ubuntu 18.04 environment
from imutils.video import VideoStream
import argparse
import imutils
import time
import dlib
import cv2

#specify file location for pre-trained landmark detector model used
parser = argparse.ArgumentParser()
parser.add_argument("-lp", "--landmark-predictor", required=True)
args = vars(parser.parse_args())

#dlib's pre-trained face detector library (based on HOG + SVM) 
face_detector = dlib.get_frontal_face_detector()
#dlib's pre-trained landmark keypoints predictor (based on gradient boosting)
landmark_predictor = dlib.shape_predictor(args["landmark_predictor"])

#read video stream from webcam
video = VideoStream(src=0).start()
time.sleep(2.0)

#indefinite loop
while True:
	#read the current frame
	curr_frame = video.read()
	#resize the frame to better detect face
	resized_frame = imutils.resize(curr_frame, width=400)
	#change to grayscale image (more efficient for face detection)
	gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
	#generates all possible bounding boxes around human faces in the image (in my case only one)
	bounding_boxes = face_detector(gray_frame, 0)

	#loop through all possible bounding box
	for bounding_box in bounding_boxes:
		#given a bounding box and current gray frame, dlib's shape_predictor method generates all landmark keypoints
		landmark_points = landmark_predictor(gray_frame, bounding_box)
		#label each keypoint as red circle dots to be displayed in real time
		for (x, y) in landmark_points:
			# arguments - frame, keypoint_location, radius, color, thickness
			cv2.circle(resized_frame, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("Realtime_Facial_Landmarks_Demo", resized_frame)
	cv2.waitKey(1)

cv2.destroyAllWindows()
video.stop()