import numpy as np
import cv2
from glob import glob
import time


infile = "C:\\Users\\maxim\\Datasets\\Datasets\\ironcar_data\\new_track\\x_chicane.npy"
outfile = "C:\\Users\\maxim\\image_raw\\"

imgs = np.load(infile)
print(imgs.shape)

for img in imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(outfile+str(time.time())+'.png',img)