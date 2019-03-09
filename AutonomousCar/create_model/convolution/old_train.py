import cv2
import numpy as np
import h5py
import time
import os

from tqdm import tqdm
from keras import callbacks
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D , MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from glob import glob

#importing custom module
import autolib

x_train= []
y_train= []

he, wi = 120, 160
dico = [3,5,7,9,11]

model_name = str(input('model name: '))

gray = False

if gray == True:
    shape = (he,wi,1)
else:
    shape = (he,wi,3)

dos= glob('C:\\Users\\maxim\\image_sorted\\*')
print(len(dos))

for img_path in dos:
    original = cv2.imread(img_path)

    #img = autolib.get_crop(original)

    img = autolib.image_process(original, color='yellow', gray= gray)


    #getting normaly and reverse label into a list
    label = autolib.get_label(img_path, before=True)

    #reshaping img
    img_shaped = np.reshape(img, shape)

    y_train.append(label[0])
    x_train.append(img_shaped)

    #flipping the image 
    img_flip = cv2.flip(img,1)

    #reshaping img_flip
    img_shaped = np.reshape(img_flip,shape)

    y_train.append(label[1])
    x_train.append(img_shaped)
    '''
    cv2.imshow('img', img+img_flip)
    cv2.waitKey(1)

cv2.destroyAllWindows()
'''
x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape, y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train/255 ,y_train, test_size =0.005)

y_train = to_categorical(y_train,len(dico)+1)
y_test = to_categorical(y_test,len(dico)+1)

#test if model_name exist, either, create a new one

try:

    model = load_model("C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\"+str(model_name))

except:
    model=Sequential()
    model.add(Conv2D(16, kernel_size=(4,4),strides=2, activation="relu", padding="same", input_shape=shape))
    model.add(MaxPooling2D(pool_size=(4,4),strides=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(len(dico)+1, activation="softmax"))

    model.summary()

model.compile(loss="categorical_crossentropy",optimizer=Adam() ,metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))


model.save('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\'+str(model_name))
