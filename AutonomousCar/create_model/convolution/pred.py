import os
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras.models import load_model

from tqdm import tqdm

import autolib


class pred():

    def __init__(self, name, path):
        self.path = path
        self.name = name

        self.img_rows = 120
        self.img_cols = 160
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        


    def get_pred(self):
        
        self.model = load_model(self.name)

        Y_test = []
        X_test = []
        confidence = []

        dos = glob(self.path)

        for i in tqdm(dos):
            img = cv2.imread(i)
            img = cv2.resize(img,(self.img_cols, self.img_rows))
            pred_img = np.reshape(img,(1, self.img_rows, self.img_cols, 3))

            #pred = autolib.image_process(img, gray=False, filter='yellow')
            pred = self.model.predict(pred_img)
            label = np.argmax(pred)

            Y_test.append(label)
            X_test.append(img)
            confidence.append(pred[0][label])

        return X_test, Y_test, confidence
        
        
        

if __name__ == "__main__":

    AI = pred(name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilter_42.h5', path = 'C:\\Users\\maxim\\image_raw\\*')

    X_test, Y_test, confidence= AI.get_pred()

    for i in tqdm(range(len(X_test))):
        img = X_test[i]
        label = Y_test[i]
        
        cv2.imwrite('C:\\Users\\maxim\\image_pred\\'+str(label)+'_'+str(time.time())+'.png',img)


