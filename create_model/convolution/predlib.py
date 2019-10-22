import os
import time
from glob import glob

import cv2
import h5py
import numpy as np
from keras.models import load_model

from tqdm import tqdm
import pandas as pd

import autolib

dire = [3,5,7,9,11]


class pred():

    def __init__(self):

        self.img_rows = 120
        self.img_cols = 160
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        


    def get_pred(self):
        
        self.model = load_model(self.name)

        dos = glob(self.path)

        X_pred = np.array([cv2.imread(i)/255 for i in tqdm(dos)])

        strt = time.time()
        preds = self.model.predict(X_pred)
        preds = np.array([np.argmax(i) for i in preds])
        end = time.time()
        print('pred in: '+str(end-strt)+' sec ||', str((end-strt)/len(X_pred))+' sec/img')

        return X_pred, preds
        

    def save_img(self):

        X_pred, Y_pred= self.get_pred()

        try:
            os.makedirs('C:\\Users\\maxim\\image_pred\\')
        except:
            pass

        for i in tqdm(range(len(X_pred))):
            img = X_pred[i]
            pred = Y_pred[i]
            label = dire[pred]
            
            cv2.imwrite('C:\\Users\\maxim\\image_pred\\'+str(label)+'_'+str(time.time())+'.png', img*255)

    

    def save_difference(self, X, Y1, Y2):


        if len(Y1) != len(Y2):
            print("Y1 don't has the same shape as Y2")
            return

        try:
            os.makedirs('C:\\Users\\maxim\\diff\\')
        except:
            pass

        for i in range(len(Y1)):
            lab1 = Y1[i]
            lab2 = Y2[i]
            
            label1 = dire[np.argmax(lab1)]
            label2 = dire[np.argmax(lab2)]

            if label1 != label2:
                img = X[i]
                cv2.imwrite('C:\\Users\\maxim\\diff\\'+str(label1)+'_'+str(label2)+'_'+str(time.time())+'.png',img)


    def confusion_table(self, Y, y):

        
        y = pd.Series(y, name='pred2')
        Y = pd.Series(np.reshape(Y,(len(Y))), name='pred1')

        df_confusion = pd.crosstab(Y, y, rownames=['pred1'], colnames=['pred2'], margins=True)
        print(df_confusion)



if __name__ == "__main__":

    AI = pred()
    AI.path = 'C:\\Users\\maxim\\image_sorted\\*'

    AI.name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilterv4_ironcar.h5'
    AI.save_img()
    
    '''
    X1, Y1 = AI.get_pred()

    AI.name = 'C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilterv5_ironcar.h5'
    X2, Y2 = AI.get_pred() 
    
    #AI.save_difference(X1, Y1, Y2)
    AI.confusion_table(Y1, Y2)
    '''
