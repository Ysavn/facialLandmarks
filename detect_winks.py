from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 11

COUNTER_LEFT = 0
COUNTER_RIGHT = 0
TOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		if leftEAR <= EYE_AR_THRESH:
			COUNTER_LEFT += 1
		elif leftEAR >= EYE_AR_THRESH:
			if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			COUNTER_LEFT = 0

		if rightEAR <= EYE_AR_THRESH:
			COUNTER_RIGHT += 1
		elif rightEAR >= EYE_AR_THRESH:
			if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			COUNTER_RIGHT = 0

		cv2.putText(frame, "Winks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()