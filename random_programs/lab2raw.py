import cv2
from tqdm import tqdm
from glob import glob
import numpy as np


path = glob('C:\\Users\\maxim\\image_sorted\\*')
save = 'C:\\Users\\maxim\\wdate\\'

for p in tqdm(path):
    img = cv2.imread(p)

    name = p.split('\\')[-1]
    lab, date = name.split('_')
    date = date.split('.png')[0]

    # cv2.imwrite(save+date+'_'+lab+'.png', img)
    # cv2.imwrite(save+date, img)
    cv2.imwrite(save+date+'.png', img)
