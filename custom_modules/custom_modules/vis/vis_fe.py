import math

import cv2
import numpy as np
from tensorflow.keras.models import Model
from . import vis_img


def visualize_fe_output(self, img,
                        input_size=(160, 120),
                        show=True, sleep_time=1):
    img = cv2.resize(img, input_size)
    fe_img = self.fe.predict(np.expand_dims(img, axis=0))[0]
    square_root = int(math.sqrt(fe_img.shape[-1]))+1

    square_img = np.zeros((square_root, square_root))
    for i in range(square_root):
        square_img[i, :] = fe_img[i*square_root:(i+1)*square_root]

    cv2.imshow('tot', square_img)
    if show:
        cv2.waitKey(sleep_time)


def visualize_model_layer_filter(model, img, layer_index,
                                 input_size=(160, 120),
                                 output_size=(80, 60),
                                 layer_outputs=None, tmp_model=None,
                                 show=True, sleep_time=0):
    img = cv2.resize(img, input_size)
    if tmp_model is None:
        if layer_outputs is None:
            layer_output = model.layers[layer_index].output
        else:
            layer_output = layer_outputs[layer_index]
        tmp_model = Model(model.input, layer_output)

    activation = tmp_model.predict(np.expand_dims(img, axis=0))[0]
    activation = np.transpose(activation, (-1, 0, 1))

    n_filter = len(activation)
    columns = int(n_filter ** 0.5)+1

    final_image = np.zeros((columns*output_size[1], columns*output_size[0]))

    for i, filter_img in enumerate(activation):
        filter_img = cv2.resize(filter_img, output_size)
        final_image[
            (i//columns)*output_size[1]:(i//columns+1)*output_size[1],
            (i % columns)*output_size[0]:(i % columns+1)*output_size[0]
        ] = filter_img

    cv2.imshow(f'{layer_index}', final_image/np.max(final_image))
    if show:
        cv2.waitKey(sleep_time)

    return activation
