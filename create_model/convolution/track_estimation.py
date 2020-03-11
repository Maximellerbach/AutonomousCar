import math
from glob import glob

import cv2
import keras
import numpy as np
from keras.models import load_model
from tqdm import tqdm

import autolib
import reorder_dataset
from architectures import dir_loss


class pos_map():
    def __init__(self, size=(512, 512), speed=-1, its=30, steer_coef=30):
        self.pmap = np.zeros((size[0], size[1], 3))
        self.speed = speed
        self.its = its
        self.dpos = self.speed*(1/self.its)
        self.steer_coef = steer_coef

        self.angle = 0
        self.pos = [0, 0]
        self.pos_list = []

    def draw(self, paths, Y):
        for ny in Y:
            average = 0
            coef = [2, 1, 0, -1, -2]

            for it, nyx in enumerate(ny):
                average += nyx*coef[it]

            degree_angle = average*self.steer_coef*(1/self.its)
            rad = math.radians(degree_angle)

            self.angle += rad
            self.angle = self.angle%(2*math.pi)

            x = math.sin(self.angle)*self.dpos # math.sin(self.dpos/self.angle)
            y = math.cos(self.angle)*self.dpos
            self.pos[0] += x
            self.pos[1] += y

            self.pos_list.append((self.pos[0], self.pos[1]))

        maxp = np.max(self.pos_list)
        minp = np.min(self.pos_list)

        h, w, _ = self.pmap.shape
        for it, p in enumerate(self.pos_list):
            rpx = ((p[0]-minp)/(maxp-minp))*w
            rpy = ((p[1]-minp)/(maxp-minp))*h
            if it==0:
                color = (0, 0, 1)
                th = 3
            else:
                color = ((it%255)/255, (255-it%255)/255, 0)
                th = 1

            cv2.circle(self.pmap, (int(rpx), int(rpy)), 1, color, thickness=th)
            img = cv2.imread(paths[it])

            cv2.imshow('img', img)
            cv2.imshow('pmap', self.pmap)
            cv2.waitKey(1)

def name_query(x):
    name = x.split('\\')[-1].split('.')[0]
    return int(name)

def load_dataset(data_path='C:\\Users\\maxim\\odo\\*'):
    paths = glob(data_path)
    paths.sort(key=name_query)

    X = np.array([cv2.resize(cv2.imread(i), (160, 120)) for i in tqdm(paths)])
    return X

def get_preds(X, model_path="test_model\\convolution\\lightv6_mix.h5"):
    model = load_model(model_path, custom_objects={"dir_loss":dir_loss})
    preds = model.predict(X)

    return preds

if __name__ == "__main__":
    # X = load_dataset()
    # Y = get_preds(X)
    dts, datalen = reorder_dataset.load_dataset('C:\\Users\\maxim\\datasets\\6\\', recursive=False)
    Y = []
    for d in dts:
        lab = autolib.get_label(d, flip=False)[0]
        Y.append(lab)

    Y = autolib.label_smoothing(Y, 5, 0) # to categorical

    pmap = pos_map()
    pmap.draw(dts, Y)
