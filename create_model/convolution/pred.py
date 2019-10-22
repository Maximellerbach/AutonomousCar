import os
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras.models import load_model

from tqdm import tqdm

import autolib

dire = [3,5,7,9,11]


class pred():

    def __init__(self):
        self.path = 'C:\\Users\\maxim\\image_raw\\*'
        self.name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilterv2_ironcar.h5'

        self.img_rows = 120
        self.img_cols = 160
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        


    def get_pred(self):
        
        self.model = load_model(self.name)

        dos = glob(self.path)

        X_pred = np.array([cv2.imread(i) for i in tqdm(dos)])
        strt = time.time()
        preds = self.model.predict(X_pred)
        end = time.time()
        print('pred in: '+str(end-strt)+' sec ||', str((end-strt)/len(X_pred))+' sec/img')

        return X_pred, preds
        

if __name__ == "__main__":

    AI = pred()

    X_pred, Y_pred= AI.get_pred()

    try:
        os.makedirs('C:\\Users\\maxim\\image_pred\\')
    except:
        pass

    for i in tqdm(range(len(X_pred))):
        img = X_pred[i]
        pred = Y_pred[i]
        label = dire[np.argmax(pred)]
        
        cv2.imwrite('C:\\Users\\maxim\\image_pred\\'+str(label)+'_'+str(time.time())+'.png',img)


