### USE ONLY IF YOU HAVE DUPLICATES AND CAN'T RECOVER IT ###
import os
import cv2
from glob import glob 
from tqdm import tqdm
import numpy as np

dataset_name = "9 sim fast"
base_dir = 'C:\\Users\\maxim\\random_data\\'
save_name = "temp"

os.mkdir(base_dir+save_name)
paths = glob(f'{base_dir}{dataset_name}\\*')

singles = [cv2.imread(paths[0])]
names = [paths[0]]
counter = 0

for path in tqdm(paths):
    img = cv2.imread(path)

    if np.mean(np.square(singles[-1]-img)) != 0.: # check mean squared error
        singles.append(img)
        names.append(path)


    if len(singles)>1:
        name = names[0].split('\\')[-1]
        cv2.imwrite(f'{base_dir}{save_name}\\{name}', singles[0])
        del singles[0]
        del names[0]
        counter += 1


print(len(paths), counter)
