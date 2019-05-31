import cv2
from tqdm import tqdm
from glob import glob


path = glob('C:\\Users\\maxim\\image_course (18-05)\\*')
save = 'C:\\Users\\maxim\\image_raw\\'

for p in tqdm(path):
    img = cv2.imread(p)

    name = p.split('\\')[-1]
    lab, date = name.split('_')

    cv2.imwrite(save+date, img)