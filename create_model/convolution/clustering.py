import collections
import os
import random
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras import callbacks
from keras.layers import (Conv2D, Dense, Dropout, Flatten, LeakyReLU, ZeroPadding2D,
                          MaxPooling2D, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose)
from keras.models import Sequential, load_model, Model, Input
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.cluster import Birch, KMeans, ward_tree
from tqdm import tqdm
import pandas as pd
import os

    
class cluster():
    def __init__(self):
        
        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        
        self.fename = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\featuresv2.h5'
        self.autoname = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\autoencoderv2.h5'
    
    
    def get_img(self, dos):

        X = []

        for i in tqdm(dos):
            img = cv2.imread(i)
            #imgflip = cv2.flip(img, 1)

            X.append(img)
            #X.append(imgflip)

        return np.array(X)


    def load_vgg(self, epochs = 10):

        try:
            fe = load_model(self.fename)
            autoencoder = load_model(self.autoname)

        except:

            inp = Input(shape=self.img_shape)

            x = Conv2D(4, kernel_size=(5,5), strides=2, activation="relu", padding="same", input_shape=self.img_shape)(inp)

            x = Conv2D(8, kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)

            x = Conv2D(16, kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)
    
            x = Conv2D(32,kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)
    
            x = Conv2D(64,kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)
            
            x = Conv2D(128,kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)

            x = Flatten()(x)

            lat = Dense(50, activation="relu")(x)

            fe = Model(inp, lat)

            lat = fe(inp)

            y = Dense(2*4*64, use_bias=False, activation="relu")(lat)

            y = Reshape((2,4,64))(y)

            y = Conv2DTranspose(64,kernel_size=(5,5), strides=2, activation="relu", padding="same")(y)
            y = ZeroPadding2D(padding=(1,1))(y)

            y = Conv2DTranspose(32,kernel_size=(5,5), strides=2, activation="relu", padding="same")(y)
            y = ZeroPadding2D(padding=(1,0))(y)

            y = Conv2DTranspose(16,kernel_size=(5,5), strides=2, activation="relu", padding="same")(y)
            y = ZeroPadding2D(padding=(1,0))(y)

            y = Conv2DTranspose(8,kernel_size=(5,5), strides=2, activation="relu", padding="same")(y)

            y = Conv2DTranspose(4,kernel_size=(5,5), strides=2, activation="relu", padding="same")(y)

            z = Conv2D(3,kernel_size=(3,3), strides=1, activation="relu", padding="same")(y)

            autoencoder = Model(inp, z)

            fe.compile(loss="mse",optimizer=Adam() ,metrics=['accuracy'])
            autoencoder.compile(loss="mse",optimizer=Adam() ,metrics=['accuracy'])

            autoencoder.summary()
        
            
        for epoch in range(1,epochs+1): #using fit but saving model every epoch

            print("epoch: %i / %i" % (epoch, epochs))
            autoencoder.fit(self.X, self.X, batch_size= self.batch_size)

            if epoch % self.save_interval == 0:
                fe.save(self.fename)
                autoencoder.save(self.autoname)
        
        return fe

    def clustering(self, dos, epochs):

        self.X = self.get_img(glob(dos))
        self.X = self.X/255

        self.img_shape = self.X[0].shape

        
        vgg = self.load_vgg(epochs=epochs)

        start = time.time()
        x = vgg.predict(self.X)        
        end = time.time()
        inter = end-start
        print(inter)


        start = time.time()
        method = Birch(threshold=2, n_clusters=None).fit(x)
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

    AI.clustering('C:\\Users\\maxim\\labelled\\*', epochs = 0)