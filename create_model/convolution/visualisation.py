from glob import glob
from random import shuffle

import cv2
import numpy as np
from keras.models import load_model
from tqdm import tqdm
from vis.backprop_modifiers import *
from vis.grad_modifiers import *
from vis.visualization import (get_num_filters, visualize_activation,
                               visualize_cam, visualize_saliency)

import architectures

model = load_model('test_model\\convolution\\lightv6_mix.h5') # , custom_objects={"dir_loss":architectures.dir_loss})
for it, i in enumerate(model.layers):
    print(i.name, it)

paths = glob('C:\\Users\\maxim\\datasets\\7 sim slow+normal\\*')
shuffle(paths)

for path in paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (160, 120))/255
    to_pred = np.expand_dims(img, axis=0)

    layer_to_study = -1
    n_filter = get_num_filters(model.layers[layer_to_study])
    preds = model.predict(to_pred)[0]
    final_hp = np.zeros((120, 160))

    for n in tqdm(range(n_filter)):
        # heatmap = visualize_cam(model, layer_to_study, n, img)
        heatmap = visualize_saliency(model, layer_to_study, n, img, backprop_modifier=guided)
        final_hp += heatmap*preds[n]

    # final_hp = np.expand_dims(final_hp, axis=-1)
    # final_hp = np.concatenate((final_hp, final_hp, final_hp), axis=-1)
    # img = img*final_hp
    average = architectures.cat2linear([preds])[0]
    print(average, preds)

    cv2.line(img, (final_hp.shape[1]//2, final_hp.shape[0]), (int(final_hp.shape[1]/2+average*30), final_hp.shape[0]-50), color=[1, 0, 0], thickness=4)
    cv2.imshow("hmp", final_hp)
    cv2.imshow("img", img)

    cv2.waitKey(0)
