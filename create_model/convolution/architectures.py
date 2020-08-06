import keras.backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.layers import *
from keras.losses import mse, mae
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed as TD


def dir_loss(y_true, y_pred):
    """
    custom loss function for the models
    (only use if you have the same models as me)
    """
    return mae(y_true, y_pred)/2 + mse(y_true, y_pred)


def linear2dir(linear, dir_range=(3, 11), to_int=True):
    delta_range = dir_range[1]-dir_range[0]
    direction = (((linear+1)/2)*delta_range)+dir_range[0]
    if to_int:
        direction = round(direction)
    return direction


def cat2linear(ny):
    averages = []
    for n in ny:
        average = 0
        coef = [-1, -0.5, 0, 0.5, 1]

        for it, nyx in enumerate(n):
            average += nyx*coef[it]
        averages.append(average)
    return averages


def create_light_CRNN(img_shape, number_class, load_fe=False, prev_act="relu", last_act="linear", drop_rate=0.1, regularizer=(0, 0), optimizer=Adam, lr=0.001, loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss], last_bias=False, load_speed=(False, False)):
    def conv_block(n_filter, kernel_size, strides, x, conv_type=Conv2D, drop=True, activation=prev_act, use_bias=False, flatten=False, batchnorm=True, padding='same'):
        x = TD(conv_type(n_filter, kernel_size=kernel_size,
                         strides=strides, use_bias=use_bias, padding=padding))(x)
        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(activation))(x)
        if drop:
            x = TD(Dropout(drop_rate))(x)
        if flatten:
            x = TD(Flatten())(x)
        return x

    def dense_block(n_neurones, x, drop=True, activation=prev_act, use_bias=False, batchnorm=True):
        x = TD(Dense(n_neurones, use_bias=use_bias))(x)
        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(activation))(x)
        if drop:
            x = TD(Dropout(drop_rate))(x)
        return x

    inputs = []

    if load_fe == True:
        fe = load_model('test_model\\convolution\\fe.h5')

    else:
        inp = Input(shape=img_shape)
        x = TD(BatchNormalization())(inp)

        x = conv_block(12, 5, 2, x, drop=False)
        x = conv_block(16, 5, 2, x, drop=False)
        x = conv_block(32, 3, 2, x, drop=True)
        x = conv_block(48, 3, 2, x, drop=False)

        x1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
        x2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        x3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        x = Concatenate()([x1, x2, x3])
        x = TD(Dropout(drop_rate))(x)

        fe = Model(inp, x)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    y = fe(inp)

    y = dense_block(150, y, batchnorm=False)
    y = dense_block(75, y, batchnorm=False)

    if load_speed[0]:
        inp = Input((img_shape[0], 1))
        inputs.append(inp)
        y = Concatenate()([y, inp])

    y = dense_block(50, y, batchnorm=False, drop=False)

    z = TD(Dense(number_class, use_bias=last_bias, activation=last_act, activity_regularizer=l1_l2(
        regularizer[0], regularizer[1]), name="steering"))(y)  # kernel_regularizer=l2(0.0005)

    if load_speed[1]:
        y = Concatenate()([y, z])
        th = TD(Dense(1, use_bias=last_bias, activation="sigmoid", activity_regularizer=l1_l2(
            regularizer[0], regularizer[1]), name="throttle"))(y)
        model = Model(inputs, [z, th])

    else:
        model = Model(inputs, z)

    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
    return model, fe


def create_light_CNN(img_shape, number_class, load_fe=False, prev_act="relu", last_act="linear", drop_rate=0.1, regularizer=(0, 0), optimizer=Adam, lr=0.001, loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss], last_bias=False, load_speed=(False, False)):
    def conv_block(n_filter, kernel_size, strides, x, conv_type=Conv2D, drop=True, activation=prev_act, use_bias=False, flatten=False, batchnorm=True, padding='same'):
        x = conv_type(n_filter, kernel_size=kernel_size,
                      strides=strides, use_bias=use_bias, padding=padding)(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if drop:
            x = Dropout(drop_rate)(x)
        if flatten:
            x = Flatten()(x)
        return x

    inputs = []

    if load_fe == True:
        fe = load_model('test_model\\convolution\\fe.h5')

    else:
        inp = Input(shape=img_shape)
        x = BatchNormalization()(inp)
        # x = GaussianNoise(0.2)(inp)

        x = conv_block(12, 5, 2, x, drop=False)
        x = conv_block(16, 5, 2, x, drop=False)
        x = conv_block(32, 3, 2, x, drop=True)
        x = conv_block(48, 3, 2, x, drop=False)

        x1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
        x2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        x3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        x = Concatenate()([x1, x2, x3])
        x = Dropout(drop_rate)(x)

        # n_out = 64+16*10+16*8
        # x = Reshape((n_out, 1))(x)
        # x = conv_block(48, n_out, n_out, x, flatten=True, drop=False, padding='valid', conv_type=Conv1D)

        ####

        fe = Model(inp, x)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    y = fe(inp)

    y = Dense(150, use_bias=False)(y)
    y = Activation(prev_act)(y)
    y = Dropout(drop_rate)(y)

    y = Dense(75, use_bias=False)(y)
    y = Activation(prev_act)(y)
    y = Dropout(drop_rate)(y)

    if load_speed[0]:
        inp = Input((img_shape[0], 1))
        inputs.append(inp)
        y = Concatenate()([y, inp])

    y = Dense(50, use_bias=False)(y)
    y = Activation(prev_act)(y)

    z = Dense(number_class, use_bias=last_bias, activation=last_act, activity_regularizer=l1_l2(
        regularizer[0], regularizer[1]), name="steering")(y)  # kernel_regularizer=l2(0.0005)

    if load_speed[1]:
        y = Concatenate()([y, z])
        th = Dense(1, use_bias=last_bias, activation="sigmoid", activity_regularizer=l1_l2(
            regularizer[0], regularizer[1]), name="throttle")(y)
        model = Model(inputs, [z, th])

    else:
        model = Model(inputs, z)

    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
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

    model.compile(loss=loss, optimizer=Adam(), metrics=metrics)

    return model, fe


def create_DepthwiseConv2D_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss]):

    inp = Input(shape=img_shape)

    x = Conv2D(32, kernel_size=1, strides=1,
               use_bias=False, padding='same')(inp)
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
    x = Conv2D(128, kernel_size=1, strides=1,
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(3, 2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=1, strides=1,
               use_bias=False, padding='same')(x)
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

    z = Dense(number_class, use_bias=False, activation=last_act,
              activity_regularizer=regularizers.l2(0.02))(y)

    model = Model(inp, z)

    model.compile(loss=loss, optimizer=Adam(), metrics=metrics)

    return model, fe


def create_lightlatent_CNN(img_shape, number_class, prev_act="relu", last_act="softmax", loss="categorical_crossentropy", metrics=["categorical_accuracy"]):

    inp = Input(shape=img_shape)

    x = Conv2D(12, 5, strides=1, use_bias=False, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = Conv2D(16, 5, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = Conv2D(24, 5, strides=1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

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

    model.compile(loss=loss, optimizer=Adam(), metrics=metrics)

    return model, fe


def flatten_model(path, save_path=None):
    model = load_model(path, custom_objects={"dir_loss": dir_loss})
    layers_dict = {"dense": Dense, "conv2d": Conv2D, "dropout": Dropout, "batch": BatchNormalization,
                   "activation": Activation, "flatten": Flatten, "zero": ZeroPadding2D, "depthwise": DepthwiseConv2D}
    inp = Input((120, 160, 3))
    x = -1

    for it, lay in enumerate(model.layers):
        lay_name = lay.name
        lay_type = lay_name.split("_")[0]

        if lay_type == "model":
            for it2, lay2 in enumerate(lay.layers):
                lay2_name = lay2.name
                lay2_type = lay2_name.split("_")[0]
                if it2 == 1 and x == -1:
                    x = layers_dict[lay2_type].from_config(
                        lay2.get_config())(inp)

                elif lay2_type in layers_dict:
                    x = layers_dict[lay2_type].from_config(
                        lay2.get_config())(x)

        if it == 1 and x == -1:
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
    new_model = create_light_CNN((120, 160, 3), 5)
