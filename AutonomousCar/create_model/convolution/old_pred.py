import cv2
import numpy as np

import h5py
from glob import glob
from tqdm import tqdm
import time

from keras.models import load_model

import autolib

os = 'win'

dico = [3,5,7,9,11]
model = str(input('model name: '))


if os == 'win':
    model = load_model("C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\"+model)
    dos = glob("C:\\Users\\maxim\\image_sorted\\*")

elif os == 'linux':
    model = load_model("/home/pi/AutonomousCar/test_model/convolution/"+model)
    dos = glob("/home/pi/image_sorted/*")

for img_path in tqdm(dos):
    img = cv2.imread(img_path)

    #img = autolib.get_crop(img)
    
    img = autolib.image_process(img, gray= False, color='yellow')
    
    image = np.expand_dims(img, axis= 0)

    predicted = model.predict(image)
    predict = dico[np.argmax(predicted)]

    '''
    #label = autolib.get_label(img_path, os = os, flip=False, before=True)
    
    if predict == label[0] or predict == 7 and label[0] == 5 or predict== 7 and label[0] == 9:
        if os == 'win':
            cv2.imwrite('C:\\Users\\maxim\\image_sorted\\'+str(predict)+'_'+str(time.time())+'.png',img)
            print('yes')
            
        if os == 'linux':
            cv2.imwrite('/home/pi/image_sorted/'+str(predict)+'_'+str(time.time())+'.png',img)
    
    cv2.imshow(str(predict),img)
    cv2.waitKey(1)
    '''

    if os == 'win':
        cv2.imwrite('C:\\Users\\maxim\\image_verif\\'+str(predict)+'_'+str(time.time())+'.png',img)
        
            
    if os == 'linux':
        cv2.imwrite('/home/pi/image_verif/'+str(predict)+'_'+str(time.time())+'.png',img)