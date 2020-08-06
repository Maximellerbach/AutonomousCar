import cv2
from tqdm import tqdm
from glob import glob
from keras.models import *
from keras.callbacks import *
import keras.backend
import skimage

import numpy as np
import autolib

path = glob('C:\\Users\\maxim\\image_raw\\*')
# model = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5')
# model.summary()

for i in path:

    img = cv2.imread(i)
    cv2.imshow('img', img)

    cv2.waitKey(1)
