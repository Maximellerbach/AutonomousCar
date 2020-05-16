from glob import glob

import cv2
import numpy as np
from keras.models import Model, load_model

model = load_model('test_model\\convolution\\fe.h5')
cut_model = Model(model.layers[0].input, model.layers[-15].output)
cut_model.summary()

dos = 'C:\\Users\\maxim\\random_data\\10 sim chicane\\*'

for path in glob(dos):
    img = cv2.imread(path)/255
    to_pred = np.expand_dims(img, axis=0)
    pred = cut_model.predict(to_pred)[0]
    pred = cv2.resize(pred, (160, 120))

    # print(pred)
    cv2.imshow('img', img)
    cv2.imshow('pred', pred)

    # cv2.imshow('pred0', pred[:, :, 0])
    # cv2.imshow('pred1', pred[:, :, 1])
    # cv2.imshow('pred2', pred[:, :, 2])

    cv2.waitKey(1)
