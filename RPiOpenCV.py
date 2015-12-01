#!/usr/bin/env python3

# import necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

# initialize camera + set res/fps + grab reference to raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 20
rawCapture = PiRGBArray(camera)

# load the required XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# allow camera to warm up
time.sleep(0.1)

# capture frames from camera
for frame in camera.capture_continuous(
    rawCapture, format="bgr", use_video_port=True):
    # grab raw NumPy array representing the image
    # use cv2.flip for horisontal flip
    image = cv2.flip(frame.array, 0)
    # load input image (or video) in grayscale mode
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Now we find the faces in the image.
# If faces are found, return positions of detected faces as Rect(x,y,w,h)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # with (x,y,w,h) set region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        # apply eye detection on this ROI (eyes are always on the face)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # show frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear stream in preparation for next frame
    rawCapture.truncate(0)

    # if 'q' key was pressed, break from loop
    if key == ord("q"):
        print("q was pressed... shutting down.")
        break
