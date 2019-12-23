import collections
import os
import random
import time
from glob import glob

import cv2
import h5py
import numpy as np
import pandas as pd
from keras import callbacks
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten, LeakyReLU,
                          MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D)
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.cluster import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from architectures import dir_loss


class cluster():
    def __init__(self):
        
        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        
        self.fename = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5'
    
    
    def get_img(self, dos):

        X = []

        for i in tqdm(dos):
            img = cv2.imread(i)
            # imgflip = cv2.flip(img, 1)

            X.append(img)
            # X.append(imgflip)

        return np.array(X)


    def load_vgg(self):
        
        fe = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5', custom_objects={"dir_loss":dir_loss})

        inp = Input(shape=(120,160,3))
        x = fe(inp)
        x = Flatten()(x)

        ffe = Model(inp, x)

        return ffe

    def clustering(self, dos, epochs):

        self.X = self.get_img(glob(dos))
        self.X = self.X/255

        self.img_shape = self.X[0].shape
        
        vgg = self.load_vgg()

        start = time.time()
        x = vgg.predict(self.X)
        end = time.time()
        inter = end-start
        print(inter)


        start = time.time()
        method = KMeans(n_clusters=30).fit(x)
        y = method.labels_
        end = time.time()
        inter = end-start
        print(inter)
        
        #Y = pd.Series(np.reshape(self.y,(len(x))), name='ground_truth')
        #y = pd.Series(y, name='pred')

        #df_confusion = pd.crosstab(Y, y, rownames=['ground_truth'], colnames=['pred'], margins=True)

        #print(df_confusion)
        
        for i in tqdm(range(len(x))):
            
            img = self.X[i]
            label = y[i]

            try:
                os.makedirs('C:\\Users\\maxim\\clustering\\'+str(label))
            except:
                pass

            cv2.imwrite('C:\\Users\\maxim\\clustering\\'+str(label)+'\\'+str(time.time())+'.png', img*255)


if __name__ == "__main__":

    AI = cluster()

    AI.save_interval = 2
    AI.batch_size = 16

    AI.clustering('C:\\Users\\maxim\\image_sorted\\*', epochs = 5)
