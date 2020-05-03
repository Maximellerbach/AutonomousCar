import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tqdm import tqdm

from data_class import Data
from scipy import spatial

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

def get_latent(self, model, path):
    img = self.load_img(path)/255
    return model.predict(np.expand_dims(img, axis=0))[0][0]

def get_latents(self, model, paths):
    latents = []
    for path in tqdm(paths):
        latents.append(get_latent(self, model, path))
    return latents

def find_nearest(a, index=0, th=0.1):
    idxs = []
    tmp_a = list(a)
    lat = tmp_a[index]
    del tmp_a[index]

    mse = np.mean(np.mean(np.abs(tmp_a-lat), axis=-1), axis=-1)
    
    for it, loss in enumerate(mse):
        if loss<th:
            if it>index:
                it += 1
            idxs.append(it)

    return idxs



if __name__ == "__main__":
    model = load_model("test_model\\convolution\\fe.h5")
    data = Data("C:\\Users\\maxim\\datasets\\", is_float=False, recursive=True)

    # new_list = []
    # for dt in data.dts:
    #     new_list += dt
    # data.dts = new_list
    data.dts = list(data.dts) # transform array to list to enable "del" operation
    
    for i in range(len(data.dts)):

        latents = get_latents(data, model, data.dts[i])
        initial_len = len(latents)

        index = 0
        del_threshold = 0.1
        while(index<len(latents)):
            nearests = find_nearest(latents, index=index)

            if len(nearests)>0:
                nearests.sort()
                nearests = list(reversed(nearests))

                for near in nearests:
                    del data.dts[i][near]
                    del latents[near]

            index += 1

        cv2.destroyAllWindows()
        print("from", initial_len, "to", len(latents))
        data.save(data.dts[i], name="light_dataset\\"+data.dts[i][0].split('\\')[-2], mode=1)
    