import collections
from glob import glob

import cv2
import keras.backend as K
import numpy as np
from keras.models import Model, load_model
from tqdm import tqdm

from architectures import dir_loss


def concatenate(img, img2, axis=0):
    shimg = img.shape
    shimg2 = img2.shape
    if axis == 0:
        img3 = np.zeros((shimg[0]+shimg2[0], shimg[1], shimg[2]))
        img3[:shimg[0] , :, :] = img
        img3[shimg[0]: , :, :] = img2
    elif axis == 1:
        img3 = np.zeros((shimg[0], shimg[1]+shimg2[1], shimg[2]))
        img3[: , :shimg[1], :] = img
        img3[: , shimg[1]:, :] = img2

    return img3

def add(img, img2):
    return img+img2

def show_filters(kernels, kshape=(5,5,3), mult=1, axis=0, kn=0):
    img = np.zeros((kshape[0]*mult, kshape[1]*mult, kshape[2]))
    tot_weights = []
    for itk, kernel in enumerate(kernels):
        visu = np.zeros(kshape)
        for itx, x in enumerate(kernel):
            for ity, y in enumerate(x):
                visu[itx, ity, :] = [y]*kshape[2]

        img = concatenate(img, cv2.resize(visu, (kshape[1]*mult, kshape[0]*mult)), axis=axis)
        # img = add(img, cv2.resize(visu, (kshape[1]*mult, kshape[0]*mult)))

    return img

def forward(model, dos, show=True, index=None):
    prev = np.zeros((256))
    m = load_model("test_model\\convolution\\lightv5_mix.h5", custom_objects={"dir_loss":dir_loss})

    inp = model.input
    outputs = [layer.output for layer in model.layers if "Conv" in str(layer)][:]

    f = K.function([inp, K.learning_phase()], outputs)

    for i in glob(dos):
        img = np.expand_dims(cv2.resize(cv2.imread(i)[320:],(160,120)), axis=0)
        layer_outs = f([img, 1.])
        
        pred = m.predict(img)[0]
        print(pred)
        # prev+=np.absolute(np.array(layer_outs).flatten())
        # print(np.array(layer_outs).flatten())

        if show==True:
            mult = 1
            for it, out in enumerate(layer_outs):
                mult += 1
                kshape = out.shape[1:]

                for j in range(out.shape[-1]):
                    visu = np.expand_dims(cv2.resize(out[0, :, :, j], (int(kshape[1]*mult), int(kshape[0]*mult))),axis=-1)

                    if j == 0:
                        img2 = visu
                    else:
                        img2 = concatenate(img2, visu, axis=1)
                    
                cv2.imshow(str(it), img2/max(img2.flatten())) # /max(img2.flatten())
                cv2.waitKey(1)

            img = img[0]
            average = 0
            coef = [-2, -1, 0, 1, 2]

            for it, nyx in enumerate(pred):
                average+=nyx*coef[it]
            cv2.line(img, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+average*30), img.shape[0]-50), color=[255, 0, 0], thickness=4)
            cv2.imshow("img", img)
            cv2.waitKey(0)

    prev = prev/max(prev)
    print(prev) # magnitude of each weight over the dataset


def get_model_weights(model):
    layers = model.layers[1].layers
    weights = [i.get_weights() for i in layers if "Conv" in str(i)]
    return weights


if __name__ == "__main__":
    model = load_model("test_model\\convolution\\fe.h5", custom_objects={"dir_loss":dir_loss})
    forward(model, 'C:\\Users\\maxim\\bdd100k\\images\\100k\\test\\*', show=True, index=1) # 'C:\Users\maxim\archive images\bdd100k\images\100k\test' 'C:\\Users\\maxim\\image_mix2\\*'

    '''
    model1 = get_model_weights(model)

    for w_lay in model1:
        print(np.array(w_lay).shape)

    for l in range(len(model1)):
        last_w = np.array(model1[l][0])
        kshape = (last_w.shape[0],last_w.shape[1],3)

        for j in range(last_w.shape[3]):
            kernels = []
            for i in range(last_w.shape[2]):
                kernels.append(last_w[:, :, i, j])

            visu = show_filters(kernels, kshape=kshape, mult=5, axis=0, kn = l)

            if j == 0:
                img = visu
            else:
                img = concatenate(img, visu, axis=1)

        cv2.imshow('k'+str(l), img/max(img.flatten())) # 
        cv2.waitKey(1)
    
    cv2.waitKey(0)
    '''