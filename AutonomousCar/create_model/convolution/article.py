from glob import glob

import cv2
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split

import autolib  # custom lib

X = []
Y = []

dos = 'C:\\Users\\maxim\\image_sorted\\*'

for img_path in glob(dos):

    img = cv2.imread(img_path)
    img_flip = cv2.flip(img, 1)

    labels = autolib.get_label(img_path, flip=True, dico=[0,1,2,3,4])

    X.append(img)
    X.append(img_flip)

    Y.append(labels[0])
    Y.append(labels[1])

X = np.array(X) / 255
Y = to_categorical(np.array(Y), 5)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

model = Sequential()

model.add(Conv2D(2, kernel_size=(5,5), use_bias=False, strides=2, activation="relu", input_shape=(120,160,3)))
model.add(Dropout(0.1))

model.add(Conv2D(4, kernel_size=(5,5), use_bias=False, strides=2, activation="relu"))
model.add(Dropout(0.1))

model.add(Conv2D(8, kernel_size=(5,5), use_bias=False, strides=2, activation="relu"))
model.add(Dropout(0.1))

model.add(Conv2D(16, kernel_size=(5,5), use_bias=False, strides=2, activation="relu"))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(32, use_bias=False, activation="relu"))
model.add(Dense(16, use_bias=False, activation="relu"))

model.add(Dense(5, use_bias=False, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer= "adam" ,metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=16, epochs=20, validation_data=(X_test,Y_test))

model.save('model.h5')