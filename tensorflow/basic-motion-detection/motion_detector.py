# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

def set(var, value):
  var = value
  return var

def get(var):
  return var

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
count = 0  
name = args.get("video")
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	ret, img = vs.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray, 1.3, 6)
# 1.3, 5 example_01
# 1.4, 5 example_03
	frame = frame if args.get("video", None) is None else frame[1]
	vidcap = frame
	text = "Unoccupied"
	face_detect = '0'
	motion_detect = '0'
	face_detect = set(face_detect,'0')
	motion_detect = set(motion_detect,'0')
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=1200)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY)[1]
#35 example_01
#50 example_03
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for (xx,yy,ww,hh) in faces:
		cv2.rectangle(frame,(xx,yy),(xx+ww,yy+hh),(255,0,0),2)
		face_detect = set(face_detect,'1')
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
		motion_detect = set(motion_detect,'1')
	if(face_detect =='1' and motion_detect =='1'):
		h,w, _ =vidcap.shape  
		output = cv2.VideoWriter( 'output.avi', fourcc,30, (w,h),True )
		output.write(vidcap)
		fname = name[7:17] + "{}.jpg".format("{0:05d}".format(count))
		print(fname)
		#print(count)
		vidcap = cv2.resize(vidcap, (640, 480), interpolation = cv2.INTER_LINEAR) 
		cv2.imwrite('image/' + fname, vidcap) 
		count += 1
		#cv2.imshow("Security ed", vidcap)
	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	#print(face_detect, motion_detect)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
output.release()
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
# python3 motion_detector.py -v videos/example_01.mp4 -a 24000

