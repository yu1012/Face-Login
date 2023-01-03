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
    embeddings = load('face_embeddings.npy')
    labels = load('face_labels.npy')


    print(len(embeddings))
    print(len(labels))
