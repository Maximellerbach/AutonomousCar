import json
import math
import time
from glob import glob

import cv2

from custom_modules import imaugm

if __name__ == "__main__":
    path = "C:\\Users\\maxim\\gen_track_user_drv_right_lane\\"
    dic = [3, 5, 7, 9, 11]

    X = []
    for i in glob(path + "*.jpg"):
        img = cv2.imread(i)
        X.append(img)

    Y = [0] * len(X)
    for i in glob(path + "*.json"):
        n_img = imaugm.get_label(i, before=False, flip=False, dico=list(range(1, len(Y) + 1)))[0]

        with open(i) as f:
            data = json.load(f)

        angle = data["user/angle"]
        angle = int((angle + 1) * 7)

        less = 11
        for it, d in enumerate(dic):
            if math.sqrt((angle - d) ** 2) < less:
                less = math.sqrt((angle - d) ** 2)
                lessit = it

        Y[n_img] = dic[lessit]

    for i in range(len(X)):
        img = X[i]
        lab = Y[i]

        cv2.imwrite("C:\\Users\\maxim\\sim_robocar\\" + str(lab) + "_" + str(time.time()) + ".png", img)
