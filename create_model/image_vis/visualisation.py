from glob import glob
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from keras import activations
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_activation

model = load_model('test_model\\models\\lightv7_mix.h5', compile=False)
model.summary()

paths = glob('C:\\Users\\maxim\\random_data\\12 sim circuit 2 new\\*.png')
shuffle(paths)
img_test = cv2.imread(paths[0])/255

layer_idx = -1

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# for filter_idx in range(5):
#     img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
#     plt.imshow(img[..., 0])
#     plt.show()

# for output_idx in np.arange(5):
#     # Lets turn off verbose output this time to avoid clutter and just see the output.
#     img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))
#     plt.figure()
#     plt.title('Networks perception of {}'.format(output_idx))
#     plt.imshow(img[..., 0])
#     plt.show()

for class_idx in range(5):
    modifier = 'guided'
    grads = visualize_saliency(model, layer_idx,
                               filter_indices=class_idx,
                               seed_input=img_test,
                               backprop_modifier=modifier,
                               grad_modifier='negate')
    plt.figure()
    plt.title(modifier)
    plt.imshow(grads, cmap='jet')
plt.show()
