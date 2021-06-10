# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :camera_test.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================

import numpy as np
import cv2
import win32gui
import os

'''
camera test file
'''
#read camera
#cap = cv2.VideoCapture(0)
video_url="rtsp://192.168.2.28:8554/live"
cap = cv2.VideoCapture(video_url)
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame = cv2.transpose(frame)

    frame = cv2.flip(frame,1)
 
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


