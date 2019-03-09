import cv2
import numpy as np

#in this program, I make custom functions

low_y = np.array([31,60,60])
up_y = np.array([50,255,255])

low_w= np.array([0,0,190])
up_w= np.array([255,10,255])

def image_process(img, size = (160,120), shape = (160,120,1), filter = True, gray = True, color = 'yellow'):
    
    img = cv2.resize(img,size)
    if filter ==True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if color == 'yellow':
            mask = cv2.inRange(hsv, low_y, up_y)
            img = cv2.bitwise_and(img, img, mask= mask)

        elif color == 'white':
            mask = cv2.inRange(hsv, low_w, up_w)
            img = cv2.bitwise_and(img, img, mask= mask)

        elif color == 'both':
            mask_y = cv2.inRange(hsv, low_y, up_y)
            mask_w = cv2.inRange(hsv, low_w, up_w)
            img = cv2.bitwise_and(img, img, mask= mask_y+mask_w)

    if gray == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img,shape)
        
    return img



def get_label(path, os = 'win', flip=True, before=True, dico=[3,5,7,9,11], rev=[11,9,7,5,3]):
    label = []

    if os == 'win':
        slash ='\\'
    elif os == 'linux':
        slash ='/'
    
    if before == True:
        name = path.split(slash)[-1]
        name = int(name.split('_')[0])
        label.append(dico.index(name))
        if flip == True:
            label.append(rev.index(name))

    if before == False:
        name = path.split('_')[-1]
        name = int(name.split('.')[0])
        label.append(dico.index(name))
        if flip == True:
            label.append(rev.index(name))
        
    return(label)

def get_crop(img, cut = 30, width= 160, height= 120):
    
    w,h = width,height-cut 
    x,y = 0,cut

    img = img[y:y+h,x:x+w]

    return(img)
