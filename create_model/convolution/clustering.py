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
    def __init__(self, fename):
        
        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        
        self.fename = fename
    
    
    def get_img(self, dos):
        X = []
        for i in tqdm(dos):
            img = cv2.imread(i)
            imgflip = cv2.flip(img, 1)
            X.append(img/255)
            # X.append(imgflip)
        return X


    def load_fe(self):
        
        fe = load_model(self.fename, custom_objects={"dir_loss":dir_loss})

        inp = Input(shape=(120,160,3))
        x = fe(inp)
        x = Flatten()(x)

        flat_fe = Model(inp, x)

        return flat_fe

    def clustering(self, doss, max_img=2000):
        paths = []
        for dos in doss:
            paths+=glob(dos)

        batchs = []
        for i in range(len(paths)//max_img):
            batchs.append(paths[i*max_img:(i+1)*max_img])

        fe = self.load_fe()
        for i, batch in tqdm(enumerate(batchs)):
            X = self.get_img(batch)
       
            x = fe.predict(np.array(X))

            if i == 0:
                method = KMeans(n_clusters=15).fit(x)
                y = method.labels_
            else:
                y = method.predict(x)

            for it in range(len(x)):
                img = X[0] # as you delete imgs, the last img will be [0]
                label = y[it]
                try:
                    os.makedirs('C:\\Users\\maxim\\clustering\\'+str(label))
                except:
                    pass
                cv2.imwrite('C:\\Users\\maxim\\clustering\\'+str(label)+'\\'+str(time.time())+'.png', img*255)

                del X[0] # clear memory



if __name__ == "__main__":

    AI = cluster('test_model\\convolution\\fe.h5')

    AI.save_interval = 2
    AI.batch_size = 16

    AI.clustering(['C:\\Users\\maxim\\datasets\\7 sim slow+normal\\*', 'C:\\Users\\maxim\\datasets\\2 donkeycar driving\\*'])# 'C:\\Users\\maxim\\recorded_imgs\\0_1587212272.138425\\*'
