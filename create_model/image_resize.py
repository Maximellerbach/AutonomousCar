import cv2
from glob import glob
import numpy as np
import os
from tqdm import tqdm

dos = glob('../../../pred_label/*.png')

for img_path in tqdm(dos):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(160,120))
    os.remove(img_path)
    cv2.imwrite(img_path,img)
