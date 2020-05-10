import cv2
import numpy as np
import random
import skimage.exposure as sk

#in this program, I make custom functions

low_y = np.array([31,60,60])
up_y = np.array([50,255,255])

low_w= np.array([0,0,190])
up_w= np.array([255,10,255])

dico=[3,5,7,9,11]
rev=[11,9,7,5,3]

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

def get_previous_label(paths, cat=True):
    """
    get the label sequence,
    return normal labels and reversed
    """
    labs = []
    revs = []
    for path in paths:
        l, r = get_label(path, cat=True)
        labs.append(l)
        revs.append(r)

    return labs, revs

def get_label(path, os_type = 'win', flip=True, cat=True, index=-1, dico=[3,5,7,9,11], rev=[11,9,7,5,3]):
    """
    get the label of an image using the patern as following: "somepath\\lab_time.png"
    """
    label = []

    if os_type == 'win':
        slash ='\\'
    elif os_type == 'linux':
        slash ='/'

    name = path.split(slash)[index].split('_')[0]
    if cat:
        lab = dico.index(int(name))
    else:
        lab = float(name)

    label.append(lab)
    if flip and cat:
        label.append(rev.index(int(name)))
    elif flip and not cat:
        label.append(-lab)

    return label

def cut_img(img, c):
    img = img[c:, :, :]
    return img

def label_smoothing(Y, n, k, random=0):
    smooth_y = []
    if random != 0:
        k = k+np.random.random()*random 
    for y in Y:
        sy = [0]*n
        sy[y] = 1-k
        if y == 2:
            sy[y-1] = k/2
            sy[y+1] = k/2

        elif y==0:
            sy[y+1] = k

        elif y==-1:
            sy[y-1] = k

        elif y==1:
            sy[y-1] = k*2/3
            sy[y+1] = k*1/3

        elif y == 3:
            sy[y-1] = k*1/3
            sy[y+1] = k*2/3

        smooth_y.append(sy)
    return np.array(smooth_y)

def change_brightness(img, lab, value=30, sign=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if sign == True:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] = v[v <= lim]+value
    if sign == False:
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] = v[v >= lim]-value

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, lab

def rescut(image, lab):
    rdm_cut = int(np.random.uniform(0, 20))
    sign = np.random.choice([True, False])

    if sign == True:
        img = image[:, rdm_cut:, :]
    else:
        dif = image.shape[1]-rdm_cut
        img = image[:, :dif, :]

    return cv2.resize(img, (160, 120)), lab

def add_random_shadow(image, lab):
    shape = image.shape
    top_y = shape[1]*np.random.uniform()
    top_x = shape[0]*np.random.uniform()
    bot_x = shape[0]*np.random.uniform()
    bot_y = shape[1]*np.random.uniform()

    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]

    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = 0.25+0.5*np.random.uniform()

    cond1 = shadow_mask==1
    cond0 = shadow_mask==0
        
    if np.random.randint(2)==1:
        image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    else:
        image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)

    return image, lab


def add_random_glow(image, lab):

    shape = image.shape
    top_y = shape[1]*np.random.uniform()
    top_x = shape[0]*np.random.uniform()
    bot_x = shape[0]*np.random.uniform()
    bot_y = shape[1]*np.random.uniform()

    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_glow = 0*image_hls[:,:,1]

    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_glow[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = 1+0.5*np.random.uniform()

    cond1 = shadow_glow==1
    cond0 = shadow_glow==0
        
    if np.random.randint(2)==1:
        image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    else:
        image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)

    return image, lab

def night_effect(img,  label, vmin=150, vmax=230):
    limit = random.uniform(vmin,vmax)
    low_limit = vmin
    int_img = sk.rescale_intensity(img, in_range=(low_limit, limit), out_range=(0,255))
    
    return int_img, label

def horizontal_flip(img, label, cat=True, dico=[3,5,7,9,11], rev=[11,9,7,5,3]):
    if cat:
        lab = dico[label]
        lab = rev.index(lab)
    else:
        lab = -label
    return cv2.flip(img, 1), lab
    
def rdm_noise(img, label):
    img = img+np.random.uniform(-25, 25, size=img.shape)
    return img, label
    
def inverse_color(img, label):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    rdm_c = np.random.uniform(0.6, 1.4, 3)
    order = [b*rdm_c[0], g*rdm_c[1], r*rdm_c[2]]
    random.shuffle(order)
    img = cv2.merge(order)
    img = img*(1/max(rdm_c))

    return img, label

def generate_random_cut(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = rescut(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug
    

def generate_brightness(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = change_brightness(X[index], Y[index], value=np.random.randint(15,45), sign=True)
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug

def generate_inversed_color(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = inverse_color(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug


def generate_low_gamma(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = change_brightness(X[index], Y[index], value=np.random.randint(15,45), sign=False)
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug

def generate_night_effect(X, Y, ys=[], proportion=0.25, ysb=False):    
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = night_effect(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug


def generate_horizontal_flip(X, Y, ys=[], proportion=0.25, ysb=False, cat=True):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = horizontal_flip(X[index], Y[index], cat=cat)
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug


def generate_random_shadows(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = add_random_shadow(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug

def generate_chained_transformations(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = change_brightness(X[index], Y[index])
            im, angle =  add_random_shadow(im, angle)
        
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug
    
def generate_random_noise(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = rdm_noise(X[index], Y[index])
        
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])
                
    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug

def generate_random_glow(X, Y, ys=[], proportion=0.25, ysb=False):
    indexes = np.random.choice([True, False], len(X), p=[proportion, 1-proportion])
    
    X_aug = []
    Y_aug = []
    YS_aug = []
    for index in range(len(X)):
        if indexes[index] == True:
            im, angle = add_random_glow(X[index], Y[index])
        
            Y_aug.append(angle)
            X_aug.append(im)
            if ysb==True:
                YS_aug.append(ys[index])

    if ysb==True:
        return X_aug, Y_aug, YS_aug
    else:
        return X_aug, Y_aug
