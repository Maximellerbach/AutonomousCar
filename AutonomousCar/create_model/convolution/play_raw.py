import cv2
from tqdm import tqdm
from glob import glob
from keras.models import *
from keras.callbacks import *
import keras.backend as K

import numpy as np
import autolib

path = glob('C:\\Users\\maxim\\image_sorted (30-11)\\*')
model = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\nofilterv2_ironcar.h5')
model.summary()

red = (0,0,255)

def get_output_layer(model, layer_name):

    # get the symbolic outputs of each "key" layer (we gave them unique names).

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer = layer_dict[layer_name]

    return layer

def visualize_class_activation_map(model, i):

        o_img = cv2.imread(i, 1)
        width, height, _ = o_img.shape

        img = np.expand_dims(o_img, axis=0)
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "max_pooling2d_4")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = (7,10))

        for i, w in enumerate(class_weights[:, 1]):
            cam += w * conv_outputs[:, :, i]

        #print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
        #heatmap[np.where(cam > 0.1)] = 0
        img = (heatmap*0.5 + o_img)/255

        return img, np.argmax(predictions)

for i in path:
    
    img, label = visualize_class_activation_map(model, i)

    #label = np.argmax(model.predict(np.expand_dims(img, axis=0)))

    cv2.imshow('img',img)
    cv2.waitKey(1)