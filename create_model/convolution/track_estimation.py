import math
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import autolib
import reorder_dataset

class pos_map():
    def __init__(self, size=(512, 512), speed=-1, its=30, steer_coef=30):
        self.pmap = np.zeros((size[0], size[1], 3))
        self.speed = speed
        self.its = its
        self.dt = 1/self.its
        self.steer_coef = steer_coef

        self.angle = 0
        self.pos = [0, 0]
        self.pos_list = []

    def draw(self, paths, Y):
        degs = []
        for p, ny in zip(paths, Y):
            average = 0
            coef = [1, 0.58, 0, -0.58, -1]

            for it, nyx in enumerate(ny):
                average += nyx*coef[it]

            # act = reorder_dataset.get_date(p)
            # dt = act-prevt
            # prevt = act

            degree_angle = average*self.steer_coef*self.dt
            rad = math.radians(degree_angle)

            self.angle += rad
            self.angle = self.angle%(2*math.pi)

            x = math.sin(self.angle)*(self.speed*self.dt) #speed is here an unknown variable
            y = math.cos(self.angle)*(self.speed*self.dt)
            self.pos[0] += x
            self.pos[1] += y

            # print(x, y, dt) # debuging

            self.pos_list.append((self.pos[0], self.pos[1]))
            degs.append(degree_angle)

        maxp = np.max(self.pos_list)
        minp = np.min(self.pos_list)
        print(minp, maxp)

        h, w, _ = self.pmap.shape
        for it, p in enumerate(self.pos_list):
            rpx = ((p[0]-minp)/(maxp-minp))*w
            rpy = ((p[1]-minp)/(maxp-minp))*h
            if it==0:
                color = (0, 0, 1)
                th = 3
            elif degs[it]>0:
                color = (degs[it]/(self.steer_coef*self.dt), 0, 0)
                th = 1
            elif degs[it]<0:
                color = (0, 0, degs[it]/(self.steer_coef*self.dt))
                th = 1
            else:
                color = (0, 1, 0)
                th = 1

            cv2.circle(self.pmap, (int(rpx), int(rpy)), 1, color, thickness=th)
            img = cv2.imread(paths[it])
            
            cv2.imshow('img', img)
            cv2.imshow('pmap', self.pmap)
            cv2.waitKey(1)

        cv2.imshow('pmap', self.pmap)
        cv2.waitKey(0)
        

def name_query(x):
    name = x.split('\\')[-1].split('.')[0]
    return int(name)

if __name__ == "__main__":
    # X = load_dataset()
    # Y = get_preds(X)
    dts, datalen = reorder_dataset.load_dataset('C:\\Users\\maxim\\datasets\\1\\', recursive=False)
    
    ts = date = reorder_dataset.get_date(dts[0])
    te = date = reorder_dataset.get_date(dts[-1])
    dt = te-ts

    datalen = len(dts)
    its = datalen/dt


    Y = []
    for d in dts:
        lab = autolib.get_label(d, flip=False)[0]
        date = reorder_dataset.get_date(d)
        Y.append(lab)

    Y = autolib.label_smoothing(Y, 5, 0) # to categorical

    pmap = pos_map(its=its, steer_coef=35, speed=1)
    pmap.draw(dts, Y)
