from os import listdir
from os.path import isdir
from numpy import load
from numpy import savez_compressed
from numpy import asarray
from PIL import Image, ImageFilter, ImageFile

import numpy as np
import face_recognition
import cv2

import os
import shutil
import time


embeddings = load('face_embeddings.npy')
labels = load('face_labels.npy')

import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


data = embeddings
count = 1
unknown_count = 0

video_capture = cv2.VideoCapture(0)

# 실시간 Face Identification
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]  # X, Y, channel(R,G,B)

    # Find all the faces and face enqcodings in the frame of video
    # 영사으로부터 얼굴 위치 탐색
    face_locations = face_recognition.face_locations(rgb_frame)

    # 탐색된 얼굴을 encoding하여 특징 추출
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
	# 탐색된 얼굴의 특징과 등록된 얼굴의 특징을 비교
        matches = face_recognition.compare_faces(data, face_encoding, tolerance=0.33)
        #여기 값을 바꿔가면서 최적의 값을 찾아야 된다

        name = "Unknown"

        max_dict = dict()
        labeled = len(matches)
        i = 0
        while (i < labeled) :
            max_dict[labels[i]] = matches[i:(i+19)].count(True)
            i += 20

        name = max(max_dict, key=max_dict.get)

        if(max_dict[name] < 2) :
            name = "Unknown"

        
        # Draw a box around the face
	# 인식된 얼굴에 Boxing하기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()