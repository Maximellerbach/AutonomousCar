import os
import random
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
import time

dicdir = [3, 5, 7, 9, 11]
dico = [48, 49, 50, 51, 52]


print(dicdir, 'from 0  to '+str(len(dicdir)-1))

dossier = 'C:\\Users\\maxim\\clustering\\*'
out_dossier = 'C:\\Users\\maxim\\labelled\\'


for dos in glob(dossier):
    frames = glob(dos+'\\*')
    print(dos)

    imgs = [cv2.imread(i) for i in frames]
    mean = np.zeros((120, 160, 3))
    for img in imgs:
        mean = mean+img

    mean = mean/(len(imgs)*255)

    while(1):
        cv2.imshow('img', mean)
        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            break
        if k in dico:
            label = dicdir[dico.index(k)]

            for img in imgs:
                cv2.imwrite(out_dossier+str(label)+'_' +
                            str(time.time())+'.png', img)
            break
        else:
            continue
