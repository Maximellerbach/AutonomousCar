import collections
import os
import random
import time
from glob import glob
from math import sqrt

import cv2
import h5py
import numpy as np
import pandas as pd
from keras import callbacks
from keras.models import Input, Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import architectures
import autolib
import predlib


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
        # model = load_model(self.name)
        # fe = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5')
        
        # model, fe = architectures.create_light_CNN((100, 160, 3), 5)
        # model, fe = architectures.create_DepthwiseConv2D_CNN(self.img_shape, 1)
        model, fe = architectures.create_heavy_CNN((100, 160, 3), 5)
        # model, fe = architectures.create_lightlatent_CNN((100, 160, 3), 5)

        fe.summary()
        model.summary()
        
        return model, fe
    

    def train(self, X=np.array([]), Y=np.array([])):
        
        #X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.1, shuffle=True)

        self.datalen = len(glob(self.path))
        self.gdos = glob(self.path)
        np.random.shuffle(self.gdos)
        self.gdos, self.valdos = np.split(self.gdos, [self.datalen-self.datalen//10])
        print(self.gdos.shape, self.valdos.shape)

        # self.img_shape = cv2.imread(self.gdos[0]).shape
        self.model, self.fe = self.build_classifier()
        
        try: #use try to early stop training if needed
            
            for epoch in range(1,self.epochs+1): #using fit but saving model every epoch

                print("epoch: %i / %i" % (epoch, self.epochs))
                #self.model.fit(X_train, Y_train, batch_size= self.batch_size, validation_data = (X_test, Y_test))
                
                self.model.fit_generator(self.image_generator(), steps_per_epoch=self.datalen//self.batch_size*2, epochs=1, validation_data=self.image_val(), validation_steps=self.datalen//10//self.batch_size)


                if epoch % self.save_interval == 0:
                    self.model.save(self.name)
                    self.fe.save('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5')
        
        except:
            print("stoped training")
            pass
        
        self.model.save(self.name)


    def get_img(self, dos):

        X = []
        Y = []

        for i in tqdm(np.sort(glob(dos))):
            img = cv2.imread(i)/255
            img = cv2.resize(img, (self.img_cols, self.img_rows))

            imgflip = cv2.flip(img, 1)

            #img = autolib.image_process(img, gray=False, filter='yellow') #yellow or white filter
            
            label = autolib.get_label(i, flip=True, before=True) # for 42's images: dico= [0,1,2,3,4], rev=[4,3,2,1,0]

            X.append(img) 
            Y.append(label[0])
            
            X.append(imgflip)
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

    def load_data(self):

        self.X, self.y = self.get_img(self.path)

        #prepare data
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.Y = to_categorical(self.y, self.number_class)
    
    def load_frames(self):
        cap = cv2.VideoCapture('C:\\Users\\maxim\\Desktop\\fh4.mp4')
    
        frames = []
        i = 0
        it = 0
        while(it<2000):
            _, frame = cap.read()
            try:
                if it %15 == 0:
                    frame = cv2.resize(frame, (320, 240))
                    frame = frame[80:,:,:]
                    frames.append(frame)
                    i += 1
                    print(i)
                it+=1
            except:
                pass

        return frames

    def image_generator(self):

        while(True):
            batchfiles = np.random.choice(self.gdos, size=self.batch_size//2)
            
            xbatch = []
            ybatch = []
            

            for i in batchfiles:
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                img = autolib.cut_img(img, 20)

                shadow = np.random.choice([True, False], p=[0.7, 0.3])
                if shadow == True:
                    img = autolib.add_random_shadow(img)

                imgflip = cv2.flip(img, 1)
                fimg = autolib.change_brightness(img, value=np.random.randint(10, 30), sign=np.random.choice([True, False]))
                ffimg = autolib.change_brightness(imgflip, value=np.random.randint(10, 30), sign=np.random.choice([True, False]))
                fimg = cv2.blur(fimg, (2,2))
                ffimg = cv2.blur(ffimg, (2,2))

                xbatch.append(fimg/255)
                xbatch.append(ffimg/255)

                label = autolib.get_label(i, flip=True, before=True, reg=False) #reg=True for [-1;1] directions
                ybatch.append(label[0])
                ybatch.append(label[1])

            xbatch = np.array(xbatch)
            ybatch = np.array(ybatch)
            ybatch = to_categorical(ybatch, self.number_class)

            yield (xbatch, ybatch)

    def image_val(self):

        while(True):
            batchfiles = np.random.choice(self.valdos, size=self.batch_size//2)
            
            xbatch = []
            ybatch = []

            for i in batchfiles:
                img = cv2.imread(i)/255
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                img = autolib.cut_img(img, 20)
                imgflip = cv2.flip(img, 1)

                label = autolib.get_label(i, flip=True, before=True, reg=False) #reg=True for [-1;1] directions

                xbatch.append(img)
                xbatch.append(imgflip)

                ybatch.append(label[0])
                ybatch.append(label[1])


            xbatch = np.array(xbatch)
            ybatch = np.array(ybatch)
            ybatch = to_categorical(ybatch, self.number_class)


            yield (xbatch, ybatch)

if __name__ == "__main__":

    AI = classifier(name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\lightv2_robo.h5', path ='C:\\Users\\maxim\\image_sorted\\*')
    #AI = classifier(name = '../../test_model/convolution/nofilterv4_ironcar.h5', path = '../../../image_raw/*')

    AI.epochs = 20
    AI.save_interval = 2
    AI.batch_size = 96

    # AI.load_data()
    AI.train()
    AI.model = load_model(AI.name)

    AI.fe = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\fe.h5')
    AI.path = 'C:\\Users\\maxim\\wdate\\*'
    AI.fe.summary()
    AI.model.summary()

    for it, i in enumerate(glob(AI.path)):
        img = cv2.imread(i)
        img = cv2.resize(img, (160, 120))
        img = autolib.cut_img(img, 20) # cut image if needed

        pred = np.expand_dims(img/255, axis=0)
        nimg = AI.fe.predict(pred)
        nimg = cv2.resize(nimg[0], (10, 2))
        nimg = np.expand_dims(nimg, axis = 0)
        st = time.time()
        ny = AI.model.predict(pred)

        lab = np.argmax(ny[0])

        et = time.time()
        dt = et-st # compute time

        n = 64

        tot_img = np.zeros((nimg.shape[1]*int(sqrt(n)), nimg.shape[2]*int(sqrt(n))))

        try:
            for x in range(int(sqrt(n))):
                for y in range(int(sqrt(n))):
                    tot_img[nimg.shape[1]*x:nimg.shape[1]*(x+1), nimg.shape[2]*y:nimg.shape[2]*(y+1)] = (nimg[0, :, :, x*int(sqrt(n))+y])
        except:
            pass

        c = np.copy(img)
        cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+(lab-2)*30), img.shape[0]-50), color=[255, 0, 0], thickness=4)
        #cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+lab*60), img.shape[0]-50), color=[255, 0, 0], thickness=4)

        # cv2.imshow('im', nimg[0, :, :])
        cv2.imshow('tot', tot_img)
        cv2.imshow('img', c/255)

        cv2.waitKey(1)

        #cv2.imwrite('C:\\Users\\maxim\\image_reg\\'+str((lab-2)/2)+'_'+str(time.time())+'.png', img)

cv2.destroyAllWindows()
