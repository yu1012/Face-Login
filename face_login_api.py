from flask import Flask
from flask import request
from flask import make_response

from numpy import load
import numpy as np
import json
import face_recognition


app = Flask(__name__)


import threading
import time
sem = threading.Semaphore()	#동시 실행을 막기 위한 semaphore


@app.route("/login", methods=['POST'])	
def login():
    sem.acquire()	#semaphore 얻어오기
    req_file = request.get_json()	#json input 기다리기
    
    face_vector = req_file['vector']	#input json file의 'vector'에 해당하는 부분
    #feature = np.array(face_vector)	#numpy array로 변환

    embeddings = load('face_embeddings.npy')
    labels = load('face_labels.npy')

    matches = face_recognition.compare_faces(embeddings, face_vector, tolerance=0.33)	#embeddings와 face_vector 비교하여 tolerance 값 기준으로 같은 얼굴인지 아닌지 판별
    name = "Unknown"	#default을 unknown으로 설정
    
    max_dict = dict()	#가장 많이 match된 label count하기 위한 dictionary
    labeled = len(matches)
    i = 0
    while (i < labeled) :
        max_dict[labels[i]] = matches[i:(i+19)].count(True)
        i += 20

    name = max(max_dict, key=max_dict.get)

    if(max_dict[name] < 2) :
        name = "Unknown"

    sem.release()

    return {'id' : name}


@app.route("/register", methods=['POST'])
def register():
    sem.acquire()
    req_file = request.get_json()
    
    name = req_file['id']
    face_vector = req_file['vector']
    feature = np.array(face_vector)

    embeddings = load('face_embeddings.npy')
    labels = load('face_labels.npy')

    embeddings = np.append(embeddings, [feature], axis=0)
    labels = np.append(labels, name)

    np.save('face_embeddings.npy', embeddings)	#배열 추가하여 덮어쓰기
    np.save('face_labels.npy', labels)	#배열 추가하여 덮어쓰기

    sem.release()
    
    return {'is_done' : True}	#vector 추가 완료되었다는 메세지 json파일로 전송

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000)	#port 5000번으로 열기


"""
input format : json
{
    'vector' : [128 float values]    //client에서 얼굴 인식 처리하여 얻은 값
}

output format : json
{
    'id' : id   //유저를 식별할 수 있는 값
                //얼굴 찾지 못할 경우, 'Unknown'으로 반환
}


@app.route("/",methods=['POST'])
def main_process():

    # heavy process here to run alone

    return "Done"
"""
