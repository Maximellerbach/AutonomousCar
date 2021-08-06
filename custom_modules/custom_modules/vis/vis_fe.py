import math

import cv2
import numpy as np
from tensorflow.keras.models import Model
import tensorflow

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.utils import normalize
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print
import tensorflow.keras.backend as K


def visualize_fe_output(self, img,
                        show=True, sleep_time=1):
    fe_img = self.fe.predict(np.expand_dims(img, axis=0))[0]
    square_root = int(math.sqrt(fe_img.shape[-1]))+1

    square_img = np.zeros((square_root, square_root))
    for i in range(square_root):
        square_img[i, :] = fe_img[i*square_root:(i+1)*square_root]

    cv2.imshow('tot', square_img)
    if show:
        cv2.waitKey(sleep_time)


def visualize_model_layer_filter(model, img, layer_index,
                                 output_size=None, mult=1,
                                 layer_outputs=None, tmp_model=None,
                                 show=True, waitkey=None):
    if tmp_model is None:
        if layer_outputs is None:
            layer_output = model.layers[layer_index].output
        else:
            layer_output = layer_outputs[layer_index]
        tmp_model = Model(model.input, layer_output)

    activation = tmp_model.predict(np.expand_dims(img, axis=0))[0]
    activation = np.transpose(activation, (-1, 0, 1))

    n_filter = len(activation)
    sqrt_filter = n_filter ** 0.5
    if int(sqrt_filter) - sqrt_filter != 0.0:
        columns = int(sqrt_filter)+1
    else:
        columns = int(sqrt_filter)

    if output_size is None:
        output_size = (activation.shape[2], activation.shape[1])
    output_size = (output_size[0]*mult, output_size[1]*mult)
    final_image = np.zeros((columns*output_size[1], columns*output_size[0]))

    for i, filter_img in enumerate(activation):
        filter_img = cv2.resize(filter_img, output_size)
        final_image[
            (i//columns)*output_size[1]:(i//columns+1)*output_size[1],
            (i % columns)*output_size[0]:(i % columns+1)*output_size[0]
        ] = filter_img

    if show:
        max_v = np.percentile(final_image, 99.9)
        # print(f'99.9 percentile of final image is :{max_v}')
        cv2.imshow(f'{layer_index}', (final_image-np.min(final_image)/max_v))
        if waitkey is not None:
            cv2.waitKey(waitkey)

    return activation, tmp_model


def get_saliency(model, inputs, class_idx, image_input_index=0, **kwargs):
    def model_modifier(m):
        m.layers[-1].activation = tensorflow.keras.activations.linear
        return m

    def loss(output):
        return output[0][class_idx]

    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=True)
    X = [tensorflow.convert_to_tensor(inp, np.float32) for inp in inputs]

    saliency_map = saliency(loss,
                            X,
                            **kwargs)
    return [normalize(saliency_map[i]) for i in range(len(saliency_map))]


def get_gradcam(model, inputs, class_idx, penultimate_layer=-1, **kwargs):  # need to work on activation modifier
    def model_modifier(m):
        m.layers[-1].activation = tensorflow.keras.activations.linear
        return m

    def loss(output):
        return output[0][class_idx]

    gradcam = Gradcam(model,
                      model_modifier=model_modifier,
                      clone=True)
    X = [tensorflow.convert_to_tensor(inp, np.float32) for inp in inputs]

    cam = gradcam(loss,
                  X,
                  penultimate_layer=penultimate_layer,
                  **kwargs)
    return [normalize(cam[i]) for i in range(len(cam))]


def vis_layer(model, layer_name, filter_number, **kwargs):
    def model_modifier(current_model):
        target_layer = current_model.get_layer(name=layer_name)
        new_model = tensorflow.keras.Model(inputs=current_model.inputs,
                                           outputs=target_layer.output)
        new_model.layers[-1].activation = tensorflow.keras.activations.linear
        return new_model

    def loss(output):
        return output[..., filter_number]

    activation_maximization = ActivationMaximization(model,
                                                     model_modifier,
                                                     clone=True)

    activation = activation_maximization(loss,
                                         callbacks=[Print(interval=50)],
                                         **kwargs)[0]
    return activation/255  # remap to [0; 1]
