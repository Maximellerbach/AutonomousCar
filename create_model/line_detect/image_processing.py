import cv2
import numpy as np
import math

def get_diff(img, prev, show=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    diff = np.absolute(gray-prev*0.99)
    diff = np.array(diff, dtype = 'uint8')

    if show == True:
        cv2.imshow('dif', diff/255)
        cv2.waitKey(1)

    return diff, gray


def get_keypoints(orb, img, cpimg, show=True):
    kps = orb.detect(img, None)
    kps, des = orb.compute(img, kps)

    if show == True:
        cpimg = cv2.drawKeypoints(cpimg, kps, cpimg, color=(0,255,0), flags=0)

        cv2.imshow('orb', cpimg/255)
        cv2.waitKey(1)

    return kps, des

def get_keypoints_distance(kp1, kp2):
    x1, y1 = kp1.pt
    x2, y2 = kp2.pt

    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def analyse_mouvement(matches, grid_shape=(16,16)):
    grid = np.array(grid_shape)