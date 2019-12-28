import keras.backend as K
from keras.regularizers import l2, l1_l2
from keras.layers import (GRU, Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Conv2DTranspose,
                          CuDNNGRU, Dense, DepthwiseConv2D, Dropout, Flatten,
                          GlobalAveragePooling2D, LeakyReLU, MaxPooling2D,
                          Reshape, SeparableConv2D, UpSampling2D, SpatialDropout2D,
                          ZeroPadding2D, concatenate, normalization)
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import SGD, Adam

def dir_loss(y_true, y_pred):
    return K.sqrt(K.square(y_true-y_pred))

def create_light_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss], recurrence=False):
    
    inp = Input(shape=img_shape)

    x = Conv2D(8, kernel_size=5, strides=3, use_bias=False, padding='valid')(inp)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    x = Dropout(0.1)(x)

    x = Conv2D(16, kernel_size=3, strides=2, use_bias=False, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)

    x = Conv2D(32, kernel_size=3, strides=2, use_bias=False, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(48, kernel_size=3, strides=2, use_bias=False, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)

    # x = Conv2D(256, kernel_size=(4,5), strides=(4,5), use_bias=False, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation(prev_act)(x)

    x = ZeroPadding2D(((1,0), 0))(x)
    x = Conv2D(256, kernel_size=1, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)
    
    x = DepthwiseConv2D(kernel_size=(5,5), strides=(5,5), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(prev_act)(x)

    x = Dropout(0.25)(x)

    ###

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)
    y = Flatten()(x)
    y = Dense(100, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(50, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
    y = Dropout(0.1)(y)

    if recurrence == True:
        inp2 = Input((49,5))
        y2 = Flatten()(inp2)
        y = concatenate([y, y2])

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dense(9, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=False, activation=last_act, activity_regularizer=l1_l2(0.01, 0.005))(y) #  kernel_regularizer=l2(0.0005)

    if recurrence == True:
        model = Model((inp, inp2), z)
    else:
        model = Model(inp, z)


    model.compile(loss=loss,optimizer=Adam(0.001) ,metrics=metrics)

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
