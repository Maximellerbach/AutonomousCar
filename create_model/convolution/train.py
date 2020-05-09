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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import *
from keras.models import Input, Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm

import architectures
import autolib
import reorder_dataset
import pred_function
# from architectures import dir_loss
from datagenerator import image_generator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

class classifier():
    def __init__(self, name, impath='', dospath='', recurrence=False, dosdir=True, memory_size=49, proportion=0.15, to_cat=True, smoothing=0, label_rdm=0):
        
        self.name = name
        self.impath = impath
        self.dospath = dospath
        self.recurrence = recurrence
        self.memory_size = memory_size
        self.dosdir = dosdir

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        self.proportion = proportion
        self.to_cat = to_cat
        self.smoothing = smoothing
        self.label_rdm = label_rdm

        self.av = []

    def build_classifier(self, model_type, load=False):
        """
        load a model using architectures program
        """
        if load == True:
            model = load_model(self.name, custom_objects={"dir_loss":architectures.dir_loss})
            fe = load_model('test_model\\convolution\\fe.h5')
        
        else:
            # model, fe = model_type((120, 160, 3), 5, loss="categorical_crossentropy", prev_act="relu", last_act="softmax", regularizer=(0.0, 0.0), lr=0.001, last_bias=True, recurrence=self.recurrence, memory=self.memory_size, metrics=["categorical_accuracy", "mse"]) # model used for the race
            model, fe = model_type((120, 160, 3), 1, loss=architectures.dir_loss, prev_act="relu", last_act="linear", regularizer=(0.0, 0.0), lr=0.001, last_bias=False, recurrence=self.recurrence, memory=self.memory_size, metrics=["mae", "mse"]) # model used for the race

            
            # model, fe = architectures.create_DepthwiseConv2D_CNN((120, 160, 3), 5)
            # model, fe = architectures.create_heavy_CNN((100, 160, 3), 5)
            # model, fe = architectures.create_lightlatent_CNN((100, 160, 3), 5)

        fe.summary()
        model.summary()

        return model, fe


    def train(self, load=False, flip=True, epochs=5, batch_size=64):
        """
        trains the model loaded as self.model
        """

        self.gdos, self.valdos, frc, self.datalen = self.get_gdos(flip=flip, cat=self.to_cat) # TODO: add folder weights
        # frc = [1]*5 # temporary to test some stuff

        print(self.gdos.shape, self.valdos.shape)
        self.model, self.fe = self.build_classifier(architectures.create_light_CNN, load=load)

        earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 0, restore_best_weights = True)

        self.model.fit_generator(image_generator(self.gdos, self.datalen, batch_size, augm=True, memory=self.memory_size, seq=self.recurrence, cat=self.to_cat, flip=flip, smoothing=self.smoothing, label_rdm=self.label_rdm), steps_per_epoch=self.datalen//(batch_size), epochs=epochs,
                                validation_data=image_generator(self.valdos, self.datalen, batch_size, augm=True, memory=self.memory_size, seq=self.recurrence, cat=self.to_cat, smoothing=self.smoothing, label_rdm=self.label_rdm), validation_steps=self.datalen//20//(batch_size),
                                class_weight=frc, callbacks=[earlystop], max_queue_size=5, workers=8)

        self.model.save(self.name)
        self.fe.save('test_model\\convolution\\fe.h5')

    def get_gdos(self, flip=True, cat=True):
        if self.recurrence == True:
            gdos, datalen = reorder_dataset.load_dataset(self.dospath)
            valdos = gdos
            if cat:
                frc = self.get_frc_cat(self.dospath, flip=flip)
            else:
                frc = [1]

        elif self.recurrence == False and self.dosdir == True:
            gdos, datalen = reorder_dataset.load_dataset(self.dospath)
            gdos = np.concatenate([i for i in gdos])
            np.random.shuffle(gdos)
            gdos, valdos = np.split(gdos, [datalen-datalen//20])
            
            if cat:
                frc = self.get_frc_cat(self.dospath+"*", flip=flip)
            else:
                frc = [1]
                
        else:
            datalen = len(glob(self.dospath))
            gdos = glob(self.dospath)
            np.random.shuffle(gdos)
            gdos, valdos = np.split(gdos, [datalen-datalen//20])
            
            if cat:
                frc = self.get_frc_cat(self.dospath, flip=flip)
            else:
                frc = [1]

        return gdos, valdos, frc, datalen

    def show_distribution(self, dos, flip=True):
        """
        calculate stats from linear labels
        and show label distribution 
        """
        return

    def get_frc_cat(self, dos, flip=True): 
        """
        calculate stats from categorical labels
        returns the weight of the classes for a balanced training
        """
        Y = []

        if self.dosdir == True:
            for d in tqdm(glob(dos)):
                for i in glob(dos+'\\*'):
                    label = autolib.get_label(i, flip=flip, cat=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

                    Y.append(label[0])
                    if flip:
                        Y.append(label[1])

        else:
            for i in tqdm(glob(dos)):
                
                label = autolib.get_label(i, flip=flip, cat=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

                Y.append(label[0])
                if flip:
                    Y.append(label[1])

        d = dict(collections.Counter(Y))
        prc = [0]*5
        l = len(Y)
        for i in range(5):
            prc[i] = d[i]/l

        frc = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
        print(frc, prc)
        return frc
    

    def calculate_FLOPS(self):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


if __name__ == "__main__":
    AI = classifier(name = 'test_model\\convolution\\linear_mix.h5', dospath ='C:\\Users\\maxim\\datasets\\*',
                    recurrence=False, dosdir=True, proportion=0.8, to_cat=False, smoothing=0.0, label_rdm=0.0) 
                    # name of the model, path to dir dataset, set dosdir for data loading, set proportion of augmented img per function

    # without augm; normally, high batch_size = better comprehension but converge less, important setting to train a CNN

    AI.train(load=False, flip=True, epochs=5, batch_size=16)
    AI.model = load_model(AI.name, compile=False) # check if the saving did well # custom_objects={"dir_loss":architectures.dir_loss}
    AI.fe = load_model('test_model\\convolution\\fe.h5')

    # print(AI.calculate_FLOPS(), "total ops")
    # iteration_speed = pred_function.evaluate_speed(AI)
    # print(iteration_speed)

    test_dos = 'C:\\Users\\maxim\\datasets\\11 sim circuit 2\\'
    pred_function.compare_pred(AI, dos=test_dos, dt_range=(0, 4000))
    pred_function.after_training_test_pred(AI, test_dos, nimg_size=(5,5), sleeptime=1)

    cv2.destroyAllWindows()