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
import matplotlib.pyplot as plt
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
import pred_function
import reorder_dataset
import dataset
# from architectures import dir_loss
from datagenerator import image_generator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

class classifier():
    def __init__(self, name, dospath='', dosdir=True, proportion=0.15, to_cat=True, sequence=False, weight_acc=0.5, smoothing=0, label_rdm=0, load_speed=(False, False)):
        
        self.Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
        self.name = name
        self.dospath = dospath
        self.dosdir = dosdir
        self.sequence = sequence

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.number_class = 5
        self.proportion = proportion
        self.to_cat = to_cat
        self.weight_acc = weight_acc
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.load_speed = load_speed

    def build_classifier(self, load=False, load_fe=False):
        """
        load a model using architectures program
        """
        if load == True:
            model = load_model(self.name, custom_objects={"dir_loss":architectures.dir_loss})
            fe = load_model('test_model\\convolution\\fe.h5')
        
        else:

            model, fe = architectures.create_light_CNN((120, 160, 3), 1, load_fe=load_fe, loss=architectures.dir_loss, 
                                    prev_act="relu", last_act="linear", drop_rate=0.15, regularizer=(0.0, 0.0), lr=0.001,
                                    last_bias=False, metrics=["mse"], load_speed=self.load_speed, sequence=True) # model used for the race
            
            # model, fe = architectures.create_DepthwiseConv2D_CNN((120, 160, 3), 5)
            # model, fe = architectures.create_heavy_CNN((100, 160, 3), 5)
            # model, fe = architectures.create_lightlatent_CNN((100, 160, 3), 5)

        fe.summary()
        model.summary()

        return model, fe


    def train(self, load=False, load_fe=False, flip=True, epochs=5, batch_size=64, seq_batchsize=64):
        """
        trains the model loaded as self.model
        """
        self.gdos, self.valdos, frc, self.datalen = self.get_gdos(flip=flip)

        print(self.gdos.shape, self.valdos.shape, self.datalen)
        self.model, self.fe = self.build_classifier(load=load, load_fe=load_fe)

        earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 0, restore_best_weights = True)

        self.model.fit_generator(image_generator(self.gdos, self.Dataset, self.datalen, batch_size, frc, load_speed=self.load_speed, sequence=self.sequence, seq_batchsize=seq_batchsize, weight_acc=self.weight_acc, augm=True, flip=flip, smoothing=self.smoothing, label_rdm=self.label_rdm), steps_per_epoch=self.datalen//(batch_size), epochs=epochs,
                                validation_data=image_generator(self.valdos, self.Dataset, self.datalen, batch_size, frc, load_speed=self.load_speed, sequence=self.sequence, seq_batchsize=seq_batchsize, weight_acc=self.weight_acc, augm=True, flip=flip, smoothing=self.smoothing, label_rdm=self.label_rdm), validation_steps=self.datalen//20//(batch_size),
                                callbacks=[earlystop], max_queue_size=5, workers=8)

        self.model.save(self.name)
        self.fe.save('test_model\\convolution\\fe.h5')

    def get_gdos(self, flip=True):
        if self.dosdir == True:
            if self.sequence:
                gdos = self.Dataset.load_dataset_sequence(self.dospath)
                gdos = np.concatenate([i for i in gdos])
                datalen = 0
                for s in gdos:
                    datalen += len(s)
            else:
                gdos = self.Dataset.load_dataset(self.dospath)
                gdos = np.concatenate([i for i in gdos])
                datalen = len(gdos)

            np.random.shuffle(gdos)
            gdos, valdos = np.split(gdos, [datalen-datalen//20])
            
        else:
            if self.sequence:
                gdos = self.Dataset.load_dos_sorted(self.dospath)
                gdos = self.Dataset.split_sorted_paths(gdos)
                datalen = 0
                for s in gdos:
                    datalen+=len(s)
            else:
                gdos = glob(self.dospath)
                datalen = len(gdos)

            np.random.shuffle(gdos)
            gdos, valdos = np.split(gdos, [datalen-datalen//20])

        if self.to_cat:
            frc = self.get_frc_cat(self.dospath, flip=flip)
        else:
            frc = self.get_frc_lin(self.dospath, flip=flip)

        return gdos, valdos, frc, datalen

    def get_frc_lin(self, dos, flip=True):
        """
        calculate stats from linear labels
        and show label distribution 
        """
        Y = []

        if self.dosdir:
            for d in tqdm(glob(dos+"*")):
                for i in glob(d+'\\*'):
                    labels = []
                    lab = self.Dataset.load_annotation(i)[0]
                    labels.append(lab)
                    if flip:
                        labels.append(-lab)

                    for l in labels: # will add normal + reversed if flip == True
                        Y.append(autolib.round_st(l, self.weight_acc))

        else:
            for i in tqdm(glob(dos+"*")):
                labels = []
                lab = self.Dataset.load_annotation(i)[0]
                labels.append(lab)
                if flip:
                    labels.append(-lab)

                for l in labels: # will add normal + reversed if flip == True
                    Y.append(autolib.round_st(l, self.weight_acc))
        d = collections.Counter(Y)

        unique = np.unique(Y)
        frc = class_weight.compute_class_weight('balanced', unique, Y)
        dict_frc = dict(zip(unique, frc))

        plt.bar(list(d.keys()), list(d.values()), width=0.2)
        plt.show()
        return dict_frc


    def get_frc_cat(self, dos, flip=True): # old, now using linear labels
        """
        calculate stats from categorical labels
        returns the weight of the classes for a balanced training
        """
        Y = []

        if self.dosdir == True:
            for d in tqdm(glob(dos)):
                for i in glob(d+'\\*'):
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
        print(prc)

        unique = np.unique(Y)
        frc = class_weight.compute_class_weight('balanced', unique, Y)
        dict_frc = dict(zip(unique, frc))

        print(dict_frc)
        return dict_frc
    

    def calculate_FLOPS(self):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


if __name__ == "__main__":
    AI = classifier(name = 'test_model\\convolution\\rnn_linearv1.h5', dospath='C:\\Users\\maxim\\datasets\\', dosdir=True, 
                    proportion=0.1, to_cat=False, sequence=False, weight_acc=2, smoothing=0.0, label_rdm=0.0, load_speed=(False, False))
                    # name of the model, path to dir dataset, set dosdir for data loading, set proportion of augmented img per function # 'C:\\Users\\maxim\\datasets\\'
                    # when weight_acc = 2, only one steering class is created

    # without augm; normally, high batch_size = better comprehension but converge less, important setting to train a CNN

    AI.train(load=False, load_fe=False, flip=True, epochs=3, batch_size=8, seq_batchsize=32)
    AI.model = load_model(AI.name, compile=False) # check if the saving did well # custom_objects={"dir_loss":architectures.dir_loss}
    AI.fe = load_model('test_model\\convolution\\fe.h5')

    # print(AI.calculate_FLOPS(), "total ops")
    # iteration_speed = pred_function.evaluate_speed(AI)
    # print(iteration_speed)

    test_dos = glob('C:\\Users\\maxim\\datasets\\*')[0]+"\\"
    # test_dos = "C:\\Users\\maxim\\random_data\\throttle\\1 ironcar driving\\"
    pred_function.compare_pred(AI, dos=test_dos, dt_range=(0, 5000))
    pred_function.speed_impact(AI, test_dos, dt_range=(0, 5000))
    pred_function.after_training_test_pred(AI, test_dos, nimg_size=(5,5), sleeptime=1)

    cv2.destroyAllWindows()
