import cv2
import numpy as np
import h5py
import time
from tqdm import tqdm
from PIL import Image
from glob import glob
import time
import os
import imutils
import datetime

i= 0
name= 0
prev= [7]
pwm= 0

wi= 30
he= 30
dossier= glob(os.path.dirname(__file__) + os.path.normpath("\\image_course\\*"))
print(len(dossier))
print(os.path.dirname(__file__) + os.path.normpath("\\image_course\\*"))
for img in tqdm(dossier):
    try:
    #if 1==1:
        cam = cv2.imread(img)
        image_tab= np.array(cam)
        hsv = cv2.cvtColor(image_tab, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,150])
        upper = np.array([60,100,255])
        
        mask = cv2.inRange(hsv, lower, upper)
        image_tab = cv2.bitwise_and(image_tab,image_tab, mask= mask)
        #cv2.imshow("img", image_tab)
        
        img= cv2.cvtColor(image_tab,cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 80, 255,0)
        im2, contours, hierarchy =cv2.findContours(thresh ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = sorted(contours, key=cv2.contourArea)
        #print(len(contours))
        
        (x,y,w,h)= cv2.boundingRect(cont[-1])
        cut = image_tab[y:y+h,x:x+w]
        cut = cv2.resize(cut,(wi,he))
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)

        date = str(datetime.datetime.now())
        date2 = date.split('-')[-1]
        save = date2.split(':')
        #cv2.imwite(os.path.dirname(__file__) +'\\image_course\\'+save[0]+'_'+save[1]+'_'+save[2]+'.png',cam)
        cv2.imwrite('..\\image_debug\\'+save[0]+'_'+save[1]+'_'+save[2]+'.png',img_gray)
        cv2.imwrite('..\\image_debugcut\\'+save[0]+'_'+save[1]+'_'+save[2]+'.png',cut)

        #cv2.imshow("img",cam)
            
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    except :
        pass    
