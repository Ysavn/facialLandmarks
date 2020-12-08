from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import playsound
from threading import Thread
import argparse
import imutils
import time
import dlib
import cv2
import simpleaudio as sa

def sound_alarm(path):
	wave_obj = sa.WaveObject.from_wave_file(path)
	play_obj = wave_obj.play()
	play_obj.wait_done()

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 90

COUNTER = 0
ALARM_ON = False

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

		ear = (leftEAR + rightEAR) / 2.0

		if ear < EYE_AR_THRESH:
			COUNTER += 1
		
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				
				if not ALARM_ON:
					ALARM_ON = True

					if args["alarm"] != "":
						t = Thread(target=sound_alarm, args=(args["alarm"],))
						t.daemon = True
						t.start()

				cv2.putText(frame, "DROWSINESS ALERT!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			COUNTER = 0
			ALARM_ON = False

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()