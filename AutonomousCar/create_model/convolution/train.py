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
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import autolib

class classifier():

    def __init__(self, name, path):
        
        self.name = name
        self.path = path

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        
        self.epochs = 10
        self.save_interval = 5
        self.batch_size = 32
            


    def build_classifier(self):

        try:
            nope
            model = load_model(self.name)

        except:

            inp = Input(shape=self.img_shape)

            x = Conv2D(2, kernel_size=(3,3), strides=1 , use_bias=False, padding='same', activation="relu", input_shape=self.img_shape)(inp)
            x = MaxPooling2D()(x)
            x = Dropout(0.2)(x)

            x = Conv2D(4, kernel_size=(3,3), strides=1, use_bias=False, padding='same', activation="relu")(x)
            x = MaxPooling2D()(x)
            x = Dropout(0.2)(x)
    
            x = Conv2D(8,kernel_size=(3,3), strides=1, use_bias=False, padding='same', activation="relu")(x)
            x = MaxPooling2D()(x)
            x = Dropout(0.2)(x)
    
            x = Conv2D(16,kernel_size=(3,3), strides=1, use_bias=False, padding='same', activation="relu")(x)
            x = MaxPooling2D()(x)
            x = Dropout(0.2)(x)

            x = Flatten()(x)

            y = Dense(32, use_bias=False, activation="relu")(x)
            y = Dense(16, use_bias=False, activation="relu")(y)

            z = Dense(self.number_class, activation="softmax")(y)

            model = Model(inp, z)

            model.compile(loss="categorical_crossentropy",optimizer=Adam() ,metrics=['accuracy'])

        model.summary()

        return model


    def train(self, X=np.array([]), Y=np.array([])):
        
        self.X, self.y = self.get_img(glob(self.path))

        #prepare data
        self.X = np.array(self.X) / 255
        self.y = np.array(self.y)
        self.Y = to_categorical(self.y, self.number_class)

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.1)

        self.img_shape = self.X[0].shape
        
        self.model = self.build_classifier()

        try: #use try to early stop training if needed
        
            for epoch in range(1,self.epochs+1): #using fit but saving model every epoch

                print("epoch: %i / %i" % (epoch, self.epochs))
                self.model.fit(X_train, Y_train, batch_size= self.batch_size, validation_data = (X_test, Y_test))

                if epoch % self.save_interval == 0:
                    self.model.save(self.name)

        except:
            print('training aborted')
            pass
        
        self.model.save(self.name)
        self.evaluate(self.model, self.X, self.y)
        


    def get_img(self, dos):

        X = []
        Y = []

        for i in tqdm(dos):
            img = cv2.imread(i)
            imgflip = cv2.flip(img, 1)

            #img = autolib.image_process(img, gray=False, filter='yellow') #yellow/white filter
            label = autolib.get_label(i,flip=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

            X.append(img)
            X.append(imgflip)

            Y.append(label[0])
            Y.append(label[1])


        counter=collections.Counter(Y) #doing basic stat
        lab_len = len(Y)
        frc = counter.most_common()
        prc = np.zeros((len(frc),2))
        
        for item in range(len(frc)):
            prc[item][0] = frc[item][0]
            prc[item][1] = frc[item][1]/lab_len*100

        print(prc) #show labels frequency in percent
        
        return X, Y


    def evaluate(self, model, X, Y):

        y = model.predict(X)
        y = [np.argmax(i) for i in tqdm(y)]

        Y = pd.Series(np.reshape(Y,(len(Y))), name='ground_truth')
        y = pd.Series(y, name='pred')

        df_confusion = pd.crosstab(Y, y, rownames=['ground_truth'], colnames=['pred'], margins=True)

        print(df_confusion)


if __name__ == "__main__":

    AI = classifier(name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilterv2_ironcar.h5', path = 'C:\\Users\\maxim\\image_sorted\\*')

    AI.epochs = 50
    AI.save_interval = 2
    AI.batch_size = 16

    AI.train()

