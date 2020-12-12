#code tested on Ubuntu 18.04 environment
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

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
seq_frames = 11

#count of continuous frames with ear below threshold for left eye
curr_count_left = 0
#count of continuous frames with ear below threshold for right eye
curr_count_right = 0
#count of overall total number of winks
total_winks = 0


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

		#eye aspect ratio for left eye
		left_ear = EAR(landmark_points[l_start:l_end])
		#eye aspect ratio for right eye
		right_ear = EAR(landmark_points[r_start:r_end])

		if left_ear < ear_threshold: 
			curr_count_left += 1
		else:
			if curr_count_left >= seq_frames: #implies a left eye wink
				total_winks += 1
			curr_count_left = 0 #reset the left eye counter

		#do same for right eye
		if right_ear < ear_threshold:
			curr_count_right += 1
		else:
			if curr_count_right >= seq_frames: #implies a right eye wink
				total_winks += 1
			curr_count_right = 0 #reset the right eye counter

		# arg: (frame, text, bottom left coordinate of text box, font, font scale factor, font color, font thickness)
		cv2.putText(resized_frame, "Winks: {}".format(total_winks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Winks_Detection_Demo", resized_frame)
	key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()