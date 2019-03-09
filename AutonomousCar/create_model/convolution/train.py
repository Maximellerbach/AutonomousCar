import os
import random
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras import callbacks
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm

import autolib


class classifier():

    def __init__(self, name, path):
        
        self.name = name
        self.path = path

        self.img_rows = 120
        self.img_cols = 160
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        self.batch_size = 32
        


    def build_classifier(self):

        try:
            model = load_model(self.name)

        except:
            model = Sequential()

            model.add(Conv2D(16, kernel_size=(3,3),strides=2, activation="relu", padding="same", input_shape=self.img_shape))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.number_class, activation="softmax"))

            model.summary()

            model.compile(loss="categorical_crossentropy",optimizer=Adam() ,metrics=["accuracy"])

        return model


    def train(self, epochs=10, X_train=np.array([]), Y_train=np.array([])):

        #prepare data
        X_train = X_train / 255
        Y_train = to_categorical(Y_train, self.number_class)

        #loss = self.model.train_on_batch(X_train,Y_train)
        for epoch in range(1,self.epochs):
            print("epoch: %i / %i" % (i, epochs) )
            self.model.fit(X_train, Y_train, batch_size= self.batch_size)

            if epoch % self.save_interval == 0:
                self.model.save(self.name)

        self.model.save(self.name)

        



    def get_img(self, dos):

        X_train = []
        Y_train = []

        for i in dos:
            img = cv2.imread(i)
            imgflip = cv2.flip(img, 1)

            #img = autolib.image_process(img, gray=False, filter='yellow')
            label = autolib.get_label(i,flip=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

            X_train.append(img)
            X_train.append(imgflip)

            Y_train.append(label[0])
            Y_train.append(label[1])

        return X_train, Y_train


if __name__ == "__main__":

    AI = classifier(name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilter.h5', path = 'C:\\Users\\maxim\\image_sorted\\*')

    AI.epochs = 10
    AI.save_interval = 1
    AI.batch_size = 128

    AI.model = AI.build_classifier()

    dos = glob(AI.path)
    X_train, Y_train = AI.get_img(dos)

    AI.train(X_train= np.array(X_train), Y_train= np.array(Y_train))

    AI.model.save(AI.name)
