from glob import glob
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tqdm import tqdm
from vis.backprop_modifiers import *
from vis.grad_modifiers import *
from vis.visualization import (get_num_filters, visualize_activation,
                               visualize_cam, visualize_saliency)

import architectures

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

model = load_model('test_model\\convolution\\lightv7_mix.h5') # , custom_objects={"dir_loss":architectures.dir_loss})
for it, i in enumerate(model.layers):
    print(i.name, it)

paths = glob('C:\\Users\\maxim\\datasets\\1 ironcar driving\\*')
shuffle(paths)

for path in paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (160, 120))/255
    to_pred = np.expand_dims(img, axis=0)

    layer_to_study = 7
    n_filter = get_num_filters(model.layers[layer_to_study])
    preds = model.predict(to_pred)[0]
    final_hp = np.zeros((120, 160))

    average = architectures.cat2linear([preds])[0]
    print(average, preds)
    cv2.line(img, (final_hp.shape[1]//2, final_hp.shape[0]), (int(final_hp.shape[1]/2+average*30), final_hp.shape[0]-50), color=[1, 0, 0], thickness=4)

    for n in tqdm(range(n_filter)):
        # heatmap = visualize_cam(model, layer_to_study, n, img)
        heatmap = visualize_saliency(model, layer_to_study, n, img, backprop_modifier=guided)

        cv2.imshow("heatmap", (heatmap/np.max(heatmap))*img)
        cv2.imshow("img", img)
        cv2.waitKey(1)

        # final_hp += heatmap*preds[n]


    # cv2.imshow("hmp", final_hp)
    # cv2.waitKey(0)
