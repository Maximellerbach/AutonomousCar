import cv2
import numpy as np
import os
from PIL import Image
from glob import glob
import time
from tqdm import tqdm

dicdir= [3,5,7,9,11]
dico= [81,82,83]

print(dicdir, 'type digit from 0  to '+str(len(dicdir))

dossier = str(input("enter filename: "))

dos= glob(dossier+"*")

for img_path in dos:
    img= cv2.imread(img_path)

    while(1):
        cv2.imshow('img',img)
        k= cv2.waitKey(0) & 0xFF
        if k == 27:
            break    
        if k in dico:
            label = dicdir[dico.index(k)]
            print(label)
            cv2.imwrite('../../new_label/'+str(label)+'_'+str(time.time())+'.png',img)
            os.remove(img_path)
            break
        else:
            continue
