from os import listdir
from os.path import isdir
from numpy import load
from numpy import savez_compressed
from numpy import asarray
from PIL import Image, ImageFilter

import numpy as np
import face_recognition
import cv2

import os
import shutil
import time



if __name__ == "__main__":
    #새로 TRAIN할 사람 이름 입력
    input_name = input()
    #파일 로딩
    embeddings = load('face_embeddings.npy')
    labels = load('face_labels.npy')

    #시작 시간 저장
    start = time.time()
    #카메라 켜고, 화면 크기 설정하기
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #저장한 횟수 세는 변수
    count = 0

    while True:
        #이미지 읽어오기
        success, image = capture.read()
        #이미지 대칭
        image = cv2.flip(image, 1)
        #얼굴 부분
        face_image = image[100:400, 500:780]
        #rgb 순서로 바꾸기
        rgb_face_image = face_image[:, :, ::-1]  # X, Y, channel(R,G,B)
        #얼굴 위치 받아오기
        face_locations = face_recognition.face_locations(rgb_face_image)

        #Face detection success
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)

            if len(face_encodings) == 1 :
                feature = np.array(face_encodings[0])
            
                embeddings = np.append(embeddings, [feature], axis = 0)
                labels = np.append(labels, input_name)
                count += 1

            cv2.rectangle(image, (480,90), (800,410), (0,255,0), 3)
        else :
            cv2.rectangle(image, (480,90), (800,410), (0,0,255), 3)
        
        cv2.imshow('LOG_IN', image)
        
        #_check_usage_of_cpu_and_memory()

        if cv2.waitKey(1) == 27 or count > 50 : break
        
        if count > 19 : break
        
        
    capture.release()
    cv2.destroyAllWindows()

    # Closes the connection
    print(time.time()-start)
    
    np.save('face_embeddings.npy', embeddings)
    np.save('face_labels.npy', labels)