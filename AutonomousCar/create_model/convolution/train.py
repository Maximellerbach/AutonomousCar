import collections
import os
import random
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras import callbacks
from keras.layers import (Conv2D, Dense, Dropout, Flatten, LeakyReLU,
                          MaxPooling2D, BatchNormalization)
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import autolib


class classifier():

    def __init__(self, name, path):
        
        self.path = 'insert here path to the training folder'
        self.name = 'AutonomousCar\\test_model\\convolution\\nofilterv3.h5' # path/name of the model you want to train/retrain

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        self.batch_size = 32

        self.save_interval = 2
        


    def build_classifier(self):

        try: #load model if it exist

            model = load_model(self.name)

        except: #create new one if it doesn't exist yet
            model = Sequential()

            model.add(Conv2D(2, kernel_size=(5,5), strides=2 , use_bias=False, activation="relu", input_shape=self.img_shape))
            model.add(Dropout(0.3))

            model.add(Conv2D(4, kernel_size=(5,5), strides=2, use_bias=False, activation="relu"))
            model.add(Dropout(0.3))

            model.add(Conv2D(8,kernel_size=(5,5), strides=2, use_bias=False, activation="relu"))
            model.add(Dropout(0.3))

            model.add(Conv2D(16,kernel_size=(5,5), strides=2, use_bias=False, activation="relu"))
            model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(32, use_bias=False, activation="relu"))
            model.add(Dense(16, use_bias=False, activation="relu"))

            model.add(Dense(self.number_class, activation="softmax"))

            model.compile(loss="categorical_crossentropy",optimizer=Adam() ,metrics=['accuracy'])
            model.summary()

        return model


    def train(self, X=np.array([]), Y=np.array([])):

        #prepare data
        X = X / 255
        Y = to_categorical(Y, self.number_class)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

        self.img_shape = X_train[0].shape

        #loss = self.model.train_on_batch(X_train,Y_train)
        for epoch in range(1,self.epochs+1): #using fit but saving model every epoch

            print("epoch: %i / %i" % (epoch, self.epochs))
            self.model.fit(X_train, Y_train, batch_size= self.batch_size, validation_data = (X_test, Y_test), verbose=1)

            if epoch % self.save_interval == 0:
                self.model.save(self.name)

        self.model.save(self.name)
        



    def get_img(self, dos):

        X_train = []
        Y_train = []

        for i in tqdm(dos):
            img = cv2.imread(i)
            imgflip = cv2.flip(img, 1)

            #img = autolib.image_process(img, gray=False, filter='yellow') #yellow/white filter
            label = autolib.get_label(i,flip=True)

            X_train.append(img)
            X_train.append(imgflip)

            Y_train.append(label[0])
            Y_train.append(label[1])


        counter=collections.Counter(Y_train) #doing basic stat
        lab_len = len(Y_train)
        frc = counter.most_common()
        prc = np.zeros((len(frc),2))
        
        for item in range(len(frc)):
            prc[item][0] = frc[item][0]
            prc[item][1] = frc[item][1]/lab_len*100

        print(prc) #show labels frequency in percent
        
        return X_train, Y_train


if __name__ == "__main__":

    AI = classifier()

    AI.epochs = 30
    AI.save_interval = 1
    AI.batch_size = 16

    AI.model = AI.build_classifier()

    dos = glob(AI.path)
    X_train, Y_train = AI.get_img(dos)

    AI.train(X= np.array(X_train), Y= np.array(Y_train))

