from keras.layers import (GRU, Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Conv2DTranspose, normalization,
                          CuDNNGRU, Dense, DepthwiseConv2D, Dropout, Flatten,
                          GlobalAveragePooling2D, LeakyReLU, MaxPooling2D,
                          Reshape, SeparableConv2D, UpSampling2D,
                          ZeroPadding2D, concatenate)
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import SGD, Adam


def create_light_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy"]):
    
    inp = Input(shape=img_shape)
            
    x = Conv2D(2, kernel_size=5, strides=2, use_bias=False, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4, kernel_size=5, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
            
    x = Conv2D(8, kernel_size=5, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, kernel_size=3, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    ###

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)
    y = Flatten()(x)

    y = Dropout(0.3)(y)
    y = Dense(35, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
    
    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
            
    y = Dense(9, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=False, activation=last_act)(y)

    model = Model(inp, z)

    model.compile(loss=loss,optimizer=Adam() ,metrics=metrics)

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

    model.compile(loss=loss,optimizer=Adam() ,metrics=metrics)

    return model, fe


def create_DepthwiseConv2D_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy"]):
    
    inp = Input(shape=img_shape)
            
    x = SeparableConv2D(2, kernel_size=5, strides=1, use_bias=False, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
            
    x = Dropout(0.05)(x)

    x = SeparableConv2D(4, kernel_size=5, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)

    x = Dropout(0.05)(x)
            
    x = SeparableConv2D(8, kernel_size=5, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)

    x = Dropout(0.05)(x)

    x = SeparableConv2D(16, kernel_size=3, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
    
    x = Dropout(0.05)(x)

    x = SeparableConv2D(24, kernel_size=3, strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(x)
    
    ###

    fe = Model(inp, x)
    inp = Input(shape=img_shape)

    x = fe(inp)
    y = Flatten()(x)

    y = Dropout(0.2)(y)
    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    y = Dense(12, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)
            
    y = Dense(9, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=False, activation=last_act)(y)

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

