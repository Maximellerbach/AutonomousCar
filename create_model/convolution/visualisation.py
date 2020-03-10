import cv2
import numpy as np
from vis.visualization import visualize_cam, visualize_saliency, visualize_activation, get_num_filters
from vis.backprop_modifiers import *
from vis.grad_modifiers import *

from keras.models import load_model

import architectures

model = load_model('test_model\\convolution\\fe.h5', custom_objects={"dir_loss":architectures.dir_loss})
img = cv2.imread('C:\\Users\\maxim\\image_sorted\\3_1570868367.5653734.png')
img = cv2.resize(img, (160,120))
img = np.expand_dims(img, axis=0)

for it, i in enumerate(model.layers):
    print(i.name, it)

layer_to_study = 19
n_filter = get_num_filters(model.layers[layer_to_study])
preds = model.predict(img)[0]
print(preds.shape)

for n in range(n_filter):
    # heatmap = visualize_cam(model, layer_to_study, n, img)
    # heatmap = visualize_activation(model, layer_to_study, filter_indices=n, act_max_weight=10, backprop_modifier=guided)
    heatmap = visualize_saliency(model, layer_to_study, n, img, backprop_modifier=guided, grad_modifier=relu)
    print(n)
    cv2.imshow("hmp", heatmap)
    cv2.waitKey(0)

