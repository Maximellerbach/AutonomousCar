import collections
import os
import random
import time
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Flatten
from keras.models import Input, Model, load_model
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn import manifold
from sklearn.cluster import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from architectures import dir_loss

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

# TODO: OPTICS/DBSCAN clustering for anomaly detection

class data_visualization():
    def __init__(self, fename):
        
        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3
        
        self.fename = fename
    
    
    def get_img(self, dos, flip=False):
        X = []
        for i in tqdm(dos):
            img = cv2.imread(i)
            X.append(img/255)
            if flip:
                imgflip = cv2.flip(img, 1)
                X.append(imgflip/255)
        return X


    def load_fe(self):
        fe = load_model(self.fename, custom_objects={"dir_loss":dir_loss})

        inp = Input(shape=(120,160,3))
        x = fe(inp)
        x = Flatten()(x)

        flat_fe = Model(inp, x)

        return flat_fe

    def get_batchs(self, doss, max_img=2000, scramble=True):
        paths = []
        for dos in doss:
            paths+=glob(dos)

        if scramble:
            random.shuffle(paths)

        batchs = []
        for i in range(len(paths)//max_img):
            batchs.append(paths[i*max_img:(i+1)*max_img])
        return batchs

    def get_latent(self, fe, X):
        return fe.predict(np.array(X))

    def get_pred(self, model, X):
        return model.predict(np.array(X))

    def normalize(self, data):
        return data/np.amax(data)

    def clustering(self, doss, max_img=2000):
        batchs = self.get_batchs(doss, max_img=max_img)
        # fe = self.load_fe()
        model = self.load_model()
        model.summary()

        for i, batch in tqdm(enumerate(batchs)):
            X = self.get_img(batch)
            X_encoded = self.get_pred(model, X)
            X_encoded = self.normalize(X_encoded)

            if i == 0:
                method = KMeans(n_clusters=3).fit(X_encoded)
                y = method.labels_
            else:
                y = method.predict(X_encoded)

            for it in range(len(X_encoded)):
                img = X[0] # as you delete imgs, the last img will be [0]
                label = y[it]
                try:
                    os.makedirs('C:\\Users\\maxim\\clustering\\'+str(label))
                except:
                    pass
                cv2.imwrite('C:\\Users\\maxim\\clustering\\'+str(label)+'\\'+str(time.time())+'.png', img*255)

                del X[0] # clear memory

    def imscatter(self, x, y, ax, imageData, zoom):
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            # Convert to image
            img = imageData[i]*255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)
        
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    def computeTSNEProjectionOfLatentSpace(self, doss, display=True): # X is here the latent representation
        batchs = self.get_batchs(doss, max_img=1000)
        model = self.load_fe()

        for batch in batchs:
            X = self.get_img(batch, flip=True)

            X_encoded = self.get_pred(model, X)

            # X_encoded = self.get_latent(fe, X)
            X_encoded = self.normalize(X_encoded)

            # print("Computing t-SNE embedding...")
            tsne = manifold.TSNE(n_components=2, init="pca", random_state=0, learning_rate=200)
            X_tsne = tsne.fit_transform(X_encoded)

            if display:
                fig, ax = plt.subplots()
                self.imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.4)
                plt.show()
            else:
                return X_tsne

if __name__ == "__main__":

    vis = data_visualization('test_model\\convolution\\fe.h5')
    doss = [i+"\\*" for i in glob('C:\\Users\\maxim\\datasets\\*')]
    vis.computeTSNEProjectionOfLatentSpace(doss, display=True)
    # vis.clustering(doss)
