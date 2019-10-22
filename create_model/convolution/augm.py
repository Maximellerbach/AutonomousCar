import cv2
import imutils
from glob import glob
import random
import time
import numpy as np
from tqdm import tqdm



def change_brightness(img, value=30, sign=True): 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if sign == True:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    if sign == False:
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def contrast(img, label): #increase or decrease image brightness
    for i in range(1,3):
        image = change_brightness(img, value= i*12, sign=True)
        image2 = change_brightness(img, value= i*12, sign=False)
        cv2.imwrite('C:\\Users\\maxim\\image_sorted\\'+str(label)+'_'+str(time.time())+'.png',image)
        cv2.imwrite('C:\\Users\\maxim\\image_sorted\\'+str(label)+'_'+str(time.time())+'.png',image2)


#dos = glob('/home/pi/image_raw/*')
dos = glob('C:\\Users\\maxim\\image_sorted\\*')

dicinv = ['11','9','7','5','3']
dicdir = ['3','5','7','9','11']


for img_path in tqdm(dos):
    img= cv2.imread(img_path)
    name = img_path.split('\\')[-1]
    value= name.split('_')[0]
    
    contrast(img,value) #augmenting normal image
    
    img = cv2.flip(img,1)
    label = dicinv[dicdir.index(value)]
    cv2.imwrite('C:\\Users\\maxim\\image_sorted\\'+str(label)+'_'+str(time.time())+'.png',img)

    contrast(img,label) #augmenting reverse image
    

print('finished')        
