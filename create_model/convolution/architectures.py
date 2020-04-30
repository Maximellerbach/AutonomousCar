import keras.backend as K
from keras.regularizers import l2, l1_l2
from keras.layers import *
from keras.losses import mse
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import SGD, Adam

def dir_loss(y_true, y_pred, wheights=np.array([-1, -0.5, 0, 0.5, 1])): # TODO: not working for the moment, need to see why (problem of shape)
    """
    custom loss function for the models
    (only use if you have the same models as me)
    """
    return K.square(y_true-y_pred) + mse(y_true, y_pred)

def linear2dir(linear, dir_range=(3, 11), to_int=True):
    delta_range = dir_range[1]-dir_range[0]
    direction = (((linear+1)/2)*delta_range)+dir_range[0]
    if to_int:
        direction = round(direction)
    return direction

def cat2linear(ny):
    averages=[]
    for n in ny:
        average = 0
        coef = [-1, -0.5, 0, 0.5, 1]

        for it, nyx in enumerate(n):
            average+=nyx*coef[it]
        averages.append(average)
    return averages

def create_light_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", regularizer=(0, 0), optimizer=Adam, lr=0.001, loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss], last_bias=False, recurrence=False, memory=49):
    
    inp = Input(shape=img_shape)
    # x = GaussianNoise(0.2)(inp)
    x = Conv2D(12, kernel_size=5, strides=2, use_bias=False, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)

    x = Conv2D(16, kernel_size=5, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)

    x = Conv2D(32, kernel_size=3, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(48, kernel_size=3, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    x = Dropout(0.1)(x)

    # x = ZeroPadding2D(((1,0), 0))(x)
    # x = DepthwiseConv2D(kernel_size=(5,5), strides=(5,5), use_bias=False, padding='same')(x)

    x = Conv2D(64, kernel_size=(8,1), strides=(8,1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    x = Dropout(0.2)(x)
    ####

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    y = fe(inp)
    y = Flatten()(y)

    y = Dense(50, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
    y = Dropout(0.1)(y)

    if recurrence == True:
        inp2 = Input((memory, 5))
        y2 = Flatten()(inp2)
        y2 = Dropout(0.2)(y2)
        
        y2 = Dense(50, use_bias=False)(y2)
        y2 = BatchNormalization()(y2)
        y2 = Activation(prev_act)(y2)

        y = concatenate([y, y2])

    # y = Dense(25, use_bias=False)(y)
    # y = BatchNormalization()(y)
    # y = Activation(prev_act)(y)

    y = Dense(9, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
    

    z = Dense(number_class, use_bias=last_bias, activation=last_act, activity_regularizer=l1_l2(regularizer[0], regularizer[1]))(y) #  kernel_regularizer=l2(0.0005)

    if recurrence == True:
        model = Model([inp, inp2], z)
    else:
        model = Model(inp, z)


    model.compile(loss=loss,optimizer=optimizer(lr=lr) ,metrics=metrics)
    return model, fe


def create_heavy_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy"]):
    
    inp = Input(shape=img_shape)
            
    x = Conv2D(12, kernel_size=5, strides=2, use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(24, kernel_size=5, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
            
    x = Conv2D(36, kernel_size=5, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(48, kernel_size=5, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, kernel_size=3, strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ###

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)
    y = Flatten()(x)

    y = Dropout(0.2)(y)
    y = Dense(100, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dropout(0.1)(y)    
    y = Dense(50, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dropout(0.1)(y)
    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dropout(0.1)(y)
    z = Dense(number_class, use_bias=False, activation=last_act)(y)

    model = Model(inp, z)

    model.compile(loss=loss, optimizer=Adam() ,metrics=metrics)

    return model, fe


def create_DepthwiseConv2D_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss]):
    
    inp = Input(shape=img_shape)
    
    
    x = Conv2D(32, kernel_size=1, strides=1, use_bias=False, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(5, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(48, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConv2D(5, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(5, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(96, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(3, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(3, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SpatialDropout2D(0.1)(x)
    x = Dropout(0.2)(x)

    ###

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)

    y = Dense(100, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dense(35, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
            
    y = Dense(9, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=False, activation=last_act, activity_regularizer=regularizers.l2(0.02))(y)

    model = Model(inp, z)

    model.compile(loss=loss,optimizer=Adam() ,metrics=metrics)

    return model, fe



def create_lightlatent_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy"]):
    
    inp = Input(shape=img_shape)
    
    x = Conv2D(12, 5, strides=1, use_bias=False, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)

    x = Conv2D(16, 5, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)

    x = Conv2D(24, 5, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)

    x = Conv2D(32, 5, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(1, 1, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)

    y = Flatten()(x)

    y = Dropout(0.2)(y)

    y = Dense(50, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
            
    y = Dense(10, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=False, activation=last_act)(y)

    model = Model(inp, z)

    model.compile(loss=loss,optimizer=Adam() ,metrics=metrics)

    return model, fe

def conv_block(x, conv, args=[8, 3, 1], activation="relu", batchnorm=True): # TODO: add beter args
    x = conv(args[0], args[1], args[2])(x)
    if batchnorm == True:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x
    
def flatten_model(path, save_path=None):
    model = load_model(path, custom_objects={"dir_loss":dir_loss})
    layers_dict = {"dense":Dense, "conv2d":Conv2D, "dropout":Dropout, "batch":BatchNormalization, "activation":Activation, "flatten":Flatten, "zero":ZeroPadding2D, "depthwise":DepthwiseConv2D}
    inp = Input((120, 160, 3))
    x = -1

    for it, lay in enumerate(model.layers):
        lay_name = lay.name
        lay_type = lay_name.split("_")[0]

        if lay_type == "model":
            for it2, lay2 in enumerate(lay.layers):
                lay2_name = lay2.name
                lay2_type = lay2_name.split("_")[0]
                if it2 == 1 and x==-1:
                    x = layers_dict[lay2_type].from_config(lay2.get_config())(inp)

                elif lay2_type in layers_dict:
                    x = layers_dict[lay2_type].from_config(lay2.get_config())(x)

        if it == 1 and x==-1:
            x = layers_dict[lay2_type].from_config(lay2.get_config())(inp)

        elif lay_type in layers_dict:
            x = layers_dict[lay_type].from_config(lay.get_config())(x)
        
    new_model = Model(inp, x)
    new_model.summary()
            
    if save_path != None:
        new_model.save(str(save_path))
    
    return new_model

if __name__ == "__main__":
    # new_model = flatten_model('test_model\\convolution\\lightv6_mix.h5')
    new_model = create_light_CNN((120,160,3), 5)