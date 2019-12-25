import collections
import os
import random
import threading
import time
from glob import glob
from math import sqrt

import cv2
import h5py
import keras
import keras.backend as K
import numpy as np
import pandas as pd
from keras import callbacks
from keras.callbacks import *
from keras.models import Input, Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm

from AutonomousCar import autolib
from AutonomousCar import architectures

import interface
import predlib
from AutonomousCar.architectures import dir_loss
from datagenerator import image_generator

class classifier():

    def __init__(self, name, impath):
        
        self.name = name
        self.impath = impath

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        
        self.epochs = 10
        self.save_interval = 5
        self.batch_size = 32

        self.av = []

    def build_classifier(self, model_type, load=False):
        """
        load a model using architectures program
        """
        if load == True:
            model = load_model(self.name, custom_objects={"dir_loss":dir_loss})
            fe = load_model('test_model\\convolution\\fe.h5')
        
        else:
            model, fe = model_type((120, 160, 3), 5, loss="categorical_crossentropy", prev_act="relu")
            
            # model, fe = architectures.create_DepthwiseConv2D_CNN((120, 160, 3), 5)
            # model, fe = architectures.create_heavy_CNN((100, 160, 3), 5)
            # model, fe = architectures.create_lightlatent_CNN((100, 160, 3), 5)

        fe.summary()
        model.summary()
        
        return model, fe


    def train(self, load=False, X=np.array([]), Y=np.array([])):
        """
        trains the model loaded as self.model
        """

        self.datalen = len(glob(self.impath))
        self.gdos = glob(self.impath)
        np.random.shuffle(self.gdos)
        self.gdos, self.valdos = np.split(self.gdos, [self.datalen-self.datalen//20])
        
        print(self.gdos.shape, self.valdos.shape)
        self.model, self.fe = self.build_classifier(architectures.create_light_CNN, load=load)

        frc = self.get_frc(self.impath)

        earlystop = EarlyStopping(monitor = 'dir_loss', min_delta = 0, patience = 3, verbose = 0, restore_best_weights = True)

        self.model.fit_generator(image_generator(self.gdos, self.batch_size, augm=True), steps_per_epoch=self.datalen//(self.batch_size), epochs=self.epochs,
                                validation_data=image_generator(self.valdos, self.batch_size, augm=True), validation_steps=self.datalen//20//(self.batch_size),
                                class_weight=frc, callbacks=[earlystop], max_queue_size=2, workers=8)
        
        # try:
        #     for epoch in range(self.epochs):
        #         self.model.fit_generator(self.image_generator(), steps_per_epoch=self.datalen//self.batch_size, epochs=1, validation_data=self.image_val(), validation_steps=self.datalen//10//self.batch_size, class_weight=frc)
        # except:
        #     print("training stopped or error occured")
        #     pass

        self.model.save(self.name)
        self.fe.save('test_model\\convolution\\fe.h5')


    def get_frc(self, dos):
        """
        calculate stats from labels
        returns the weight of the classes for balanced training
        """
        Y = []
        for i in tqdm(np.sort(glob(dos))):
            
            label = autolib.get_label(i, flip=True, before=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

            Y.append(label[0])
            # if label[1] != 2:
            Y.append(label[1])

        d = dict(collections.Counter(Y))
        prc = [0]*5
        l = len(Y)
        for i in range(5):
            prc[i] = d[i]/l

        frc = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
        print(frc, prc)
        return frc
    
    def load_frames(self, path, size=(360,240), batch_len=32):
        """
        load a batch of frame from video
        """
        batch = []
        
        for _ in range(batch_len):
            _, frame = self.cap.read()
            frame = cv2.resize(frame, size)
            batch.append(frame)

        return batch

    def pred_img(self, img , size, cut, sleeptime, n, nimg_size=(20, 15)):
        
        """
        predict an image and visualize the prediction
        """

        img = autolib.cut_img(img, cut) # cut image if needed
        img = cv2.resize(img, size)
        pred = np.expand_dims(img/255, axis=0)
        nimg = AI.fe.predict(pred)
        nimg = np.expand_dims(cv2.resize(nimg[0], nimg_size), axis=0)

        ny = AI.model.predict(pred)[0]
        lab = np.argmax(ny)
        
        # average softmax direction
        average = 0
        coef = [-2, -1, 0, 1, 2]

        for it, nyx in enumerate(ny):
            average+=nyx*coef[it]

        
        if len(self.av)<1:
            self.av.append(average)
        else:
            self.av.insert(0, average)
            del self.av[-1]

        avaverage = np.average(self.av)


        ny = [round(n, 3) for n in ny]
        tot_img = np.zeros((nimg.shape[1]*int(sqrt(n)), nimg.shape[2]*int(sqrt(n))))

        try:
            for x in range(int(sqrt(n))):
                for y in range(int(sqrt(n))):
                    tot_img[nimg.shape[1]*x:nimg.shape[1]*(x+1), nimg.shape[2]*y:nimg.shape[2]*(y+1)] = (nimg[0, :, :, x*int(sqrt(n))+y])
        except:
            pass

        c = np.copy(img)
        cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+avaverage*30), img.shape[0]-50), color=[255, 0, 0], thickness=4)
        # cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+average*30), img.shape[0]-50), color=[0, 0, 255], thickness=4)
        c = c/255

        if n==1:
            av = nimg[0]
            av = cv2.resize(av, size)
            cv2.imshow('im', nimg[0, :, :])
            
        else:
            av = np.sum(nimg[0], axis=-1)
            av = cv2.resize(av/(nimg.shape[-1]/2), size)
            cv2.imshow('tot', tot_img)


        cv2.imshow('av', av*c[:,:,0])
        cv2.imshow('img', c)

        cv2.waitKey(sleeptime)


    def after_training_test_pred(self, path, size, cut=0, n=9, nimg_size=(20,15), from_path=True, from_vid=False, batch_vid=32, sleeptime=1):
        """
        either predict images in a folder
        or from a video
        """

        if from_path==True:
            for it, i in enumerate(glob(path)):
                img = cv2.imread(i)
                img = cv2.resize(img, size)
                # img, _ = autolib.rdm_noise(img, 0)
                # img, _ = autolib.night_effect(img, 0)
                self.pred_img(img, size, cut, sleeptime, n, nimg_size=nimg_size)
                
        elif from_vid==True:
            self.cap = cv2.VideoCapture(path)
            print("created cap")

            while(True):
                im_batch = self.load_frames(path, batch_len=batch_vid)
                for img in im_batch:
                    self.pred_img(img, size, cut, sleeptime, n)


if __name__ == "__main__":
    AI = classifier(name = 'test_model\\convolution\\lightv4_mix.h5', impath ='C:\\Users\\maxim\\image_mix2\\*.png')

    AI.epochs = 5
    AI.save_interval = 2
    AI.batch_size = 48

    AI.train(load=True)
    AI.model = load_model(AI.name, custom_objects={"dir_loss":dir_loss})

    AI.fe = load_model('test_model\\convolution\\fe.h5')
    AI.after_training_test_pred('C:\\Users\\maxim\\wdate\\*', (160,120), cut=0, from_path=True, from_vid=False, n=256, nimg_size=(4,4), sleeptime=1)
    # AI.after_training_test_pred('F:\\fh4.mp4', (160,120), cut=100, from_path=False, from_vid=True, n=49, batch_vid=1)

    cv2.destroyAllWindows()
