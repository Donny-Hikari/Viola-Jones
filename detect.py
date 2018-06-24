# 
# detect.py
#   A face detection example using FaceDetector and mergeRects.
# 
# Author : Donny
# 

import cv2
import shutil
import os
import numpy as np
import scipy as sp
import scipy.misc as spmisc
from facedetector import FaceDetector
from mergerect import mergeRects

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    faceDetector = FaceDetector()

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces, totalTiles = faceDetector.detect(
            gray,
            min_size=0.0, max_size=0.3,
            step=0.9, detectPad=(2,2),
            verbose=True,
            getTotalTiles=True
        )
        faces = mergeRects(
            faces,
            overlap_rate=0.82,
            min_overlap_cnt=4
        )

        print("----faces detected----")
        for x, y, w, h in faces:
            print(x, y, w, h)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        frame = cv2.resize(frame, (0,0), fx=1.0/0.4, fy=1.0/0.4)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    faceDetector.stopParallel()
    video_capture.release()
    cv2.destroyAllWindows()
