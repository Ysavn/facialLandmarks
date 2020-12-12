#code tested on Ubuntu 18.04 environment
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import imutils
import time
import dlib
import cv2
import simpleaudio as sa

#make an alert sound to wake user from their drowsiness
def alert_sound():
	wave_obj = sa.WaveObject.from_wave_file('./alert_sound.wav')
	play_obj = wave_obj.play()
	play_obj.wait_done()

#calculate eye aspect ratio (EAR)
def EAR(eye_keypoints):
	#calculate euclidean distance between first pair of keypoints along the width of eye
	width1 = dist.euclidean(eye_keypoints[1], eye_keypoints[5])
	#calculate euclidean distance between second pair of keypoints along the width of eye
	width2 = dist.euclidean(eye_keypoints[2], eye_keypoints[4])
	#calculate euclidean distance between the only pair of keypoints along the length of eye
	length = dist.euclidean(eye_keypoints[0], eye_keypoints[3])
	#calculating the aspect ratio (i.e. width/height)
	#height multiplied by 2 here correponding to 2 estimates of width
	return (width1 + width2) / (length * 2.0)

#EAR threshold
ear_threshold = 0.18
#Number of continuous frames with EAR below threshold => eye blink
seq_frames = 90

#count of continuous frames with ear below threshold
curr_count = 0
#whether alert sound is active or not
alert_on = False

#dlib's pre-trained face detector library (based on HOG + SVM) 
face_detector = dlib.get_frontal_face_detector()
#dlib's pre-trained landmark keypoints predictor (based on gradient boosting)
landmark_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#getting start and end keypoint indexes from the facial landmarks map
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
		#convert to numpy array (allows indexing)
		landmark_points = face_utils.shape_to_np(landmark_points)

		#calculate eye aspect ratio for both eyes and average them out to reduce noise
		avg_ear = (EAR(landmark_points[l_start:l_end]) + EAR(landmark_points[r_start:r_end]))/2.0

		if avg_ear < ear_threshold:
			curr_count += 1
		
			if curr_count >= seq_frames: #implies prolonged eyes closed, hence need to alert user for drowsiness
				if not alert_on:
					alert_on = True
					t = Thread(target=alert_sound, args=()) #spawn a thread to generate alert sound in parallel
					t.daemon = True #allows the thread to run in background
					t.start()

				# arg: (frame, text, bottom left coordinate of text box, font, font scale factor, font color, font thickness)
				cv2.putText(resized_frame, "DROWSINESS ALERT!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			curr_count = 0 #user alert, so reset the counter
			alert_on = False #no need for alert sound hence

	cv2.imshow("Drowsiness_Check_Demo", resized_frame)
	key = cv2.waitKey(1)

cv2.destroyAllWindows()
vs.stop()