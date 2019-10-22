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



def get_label(path, os = 'win', flip=True, before=True, reg=False, dico=[3,5,7,9,11], rev=[11,9,7,5,3]):
    label = []

    if os == 'win':
        slash ='\\'
    elif os == 'linux':
        slash ='/'

    if reg == False:
        
        if before == True:
            lab = path.split(slash)[-1]
            lab = int(lab.split('_')[0])
            label.append(dico.index(lab))
            if flip == True:
                label.append(rev.index(lab))

        if before == False:
            lab = path.split('_')[-1]
            lab = int(lab.split('.')[0])
            label.append(dico.index(lab))
            if flip == True:
                label.append(rev.index(lab))
    
    elif reg == True:

        if before == True:
            lab = path.split(slash)[-1]
            lab = lab.split('_')[0]
            lab = float(lab)
            label.append(lab)
            if flip == True:
                label.append(-lab)

        if before == False:
            lab = path.split('_')[-1]
            lab = lab.split('.')[0]
            lab = float(lab)
            label.append(lab)
            if flip == True:
                label.append(-lab)

    
    return label

def cut_img(img, c):
    img = img[c:, :, :]
    return img


def get_augm(img):
    imgs = []

    for i in [[20, False], [20, True]]:
        n = change_brightness(img, i[0], i[1])
        imgs.append(n)

    return imgs


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


def add_random_shadow(image):

    shape = image.shape
    top_y = shape[1]*np.random.uniform()
    top_x = 0
    bot_x = shape[0]
    bot_y = shape[1]*np.random.uniform()

    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]

    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()

    random_bright = .5

    cond1 = shadow_mask==1
    cond0 = shadow_mask==0
        
    if np.random.randint(2)==1:
        image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    else:
        image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)

    return image


if __name__ == "__main__":
    #few tests
    from glob import glob
    dos = glob('C:\\Users\\maxim\\image_reg\\*')
    print(dos)
    
    for i in dos:
        im = cv2.imread(i)
        lab = get_label(i, flip=True, before=True, reg=True)
        print(lab)
        cv2.imshow(str(lab[0]), im)
        cv2.waitKey(0)

