from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, DepthwiseConv2D, Dropout, Flatten,
                          ZeroPadding2D, MaxPooling2D)
from keras.layers.wrappers import TimeDistributed as TD
from keras.losses import mae, mse
from keras.models import Input, Model, Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l1_l2


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


def create_light_CRNN(dataset, img_shape, load_fe=False,
                      prev_act="relu", last_act="linear", padding="same",
                      drop_rate=0.1, use_bias=False, regularizer=(0, 0), optimizer=Adam, lr=0.001,
                      loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss],
                      input_components=[], output_components=[]):

    def conv_block(n_filter, kernel_size, strides, x,
                   conv_type=Conv2D, drop=True,
                   activation=prev_act, use_bias=use_bias,
                   flatten=False, batchnorm=True, padding=padding):

        x = TD(conv_type(
            n_filter, kernel_size=kernel_size,
            strides=strides, use_bias=use_bias, padding=padding,
            kernel_regularizer=l1_l2(regularizer[0], regularizer[1]),
            bias_regularizer=l1_l2(regularizer[0], regularizer[1])))(x)
        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(activation))(x)
        if drop:
            x = TD(Dropout(drop_rate))(x)
        if flatten:
            x = TD(Flatten())(x)
        return x

    def dense_block(n_neurones, x,
                    drop=True, activation=prev_act,
                    use_bias=use_bias, batchnorm=True):
        x = TD(Dense(n_neurones, use_bias=use_bias))(x)
        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(activation))(x)
        if drop:
            x = TD(Dropout(drop_rate))(x)
        return x

    inputs = []

    if load_fe:
        fe = load_model('test_model\\models\\fe.h5')

    else:
        inp = Input(shape=img_shape)
        x = TD(BatchNormalization())(inp)

        x = conv_block(12, 5, 2, x, drop=True)
        x = conv_block(16, 5, 2, x, drop=True)
        x = conv_block(32, 3, 2, x, drop=True)
        x = conv_block(48, 3, 2, x, drop=True)

        x1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
        x2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        x3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        x = Concatenate()([x1, x2, x3])
        x = TD(Dropout(drop_rate))(x)

        fe = Model(inp, x)

    inputs = []
    outputs = []
    input_components_names = dataset.indexes2components_names(input_components)
    output_components_names = dataset.indexes2components_names(
        output_components)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    y = fe(inp)

    y = dense_block(150, y, batchnorm=False)
    y = dense_block(75, y, batchnorm=False)

    if 'speed' in input_components_names:
        inp = Input((img_shape[0], 1))
        inputs.append(inp)
        y = Concatenate()([y, inp])

    y = dense_block(50, y, batchnorm=False, drop=False)

    if 'direction' in input_components_names:
        z = Dense(1, use_bias=use_bias, activation=last_act, activity_regularizer=l1_l2(
            regularizer[0], regularizer[1]), name="steering")(y)  # kernel_regularizer=l2(0.0005)
        outputs.append(z)
        y = Concatenate()([y, z])

    if 'throttle' in output_components_names:
        th = Dense(1, use_bias=use_bias, activation="sigmoid", activity_regularizer=l1_l2(
            regularizer[0], regularizer[1]), name="throttle")(y)
        outputs.append(th)
        y = Concatenate()([y, th])

    model = Model(inputs, outputs)
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
    return model, fe


def create_light_CNN(dataset, img_shape, load_fe=False,
                     prev_act="relu", last_act="linear", padding='same',
                     drop_rate=0.1, use_bias=False, regularizer=(0, 0), optimizer=Adam, lr=0.001,
                     loss="categorical_crossentropy", metrics=["categorical_accuracy", dir_loss],
                     input_components=[], output_components=[]):

    def conv_block(n_filter, kernel_size, strides, x,
                   conv_type=Conv2D, drop=True,
                   activation=prev_act, use_bias=use_bias,
                   flatten=False, batchnorm=True, padding=padding):

        x = conv_type(
            n_filter, kernel_size=kernel_size,
            strides=strides, use_bias=use_bias, padding=padding,
            kernel_regularizer=l1_l2(regularizer[0], regularizer[1]),
            bias_regularizer=l1_l2(regularizer[0], regularizer[1]))(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if drop:
            x = Dropout(drop_rate)(x)
        if flatten:
            x = Flatten()(x)
        return x

    def dense_block(n_neurones, x,
                    drop=True, activation=prev_act,
                    use_bias=use_bias, batchnorm=True):
        x = Dense(n_neurones, use_bias=use_bias,
                  kernel_regularizer=l1_l2(regularizer[0], regularizer[1]),
                  bias_regularizer=l1_l2(regularizer[0], regularizer[1]))(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if drop:
            x = Dropout(drop_rate)(x)
        return x

    if load_fe:
        fe = load_model('test_model\\models\\fe.h5')

    else:
        inp = Input(shape=img_shape)
        x = BatchNormalization()(inp)
        # x = GaussianNoise(0.2)(inp)

        # x = conv_block(16, 5, 2, x, drop=True)
        # x = conv_block(16, 5, 2, x, drop=True)
        # x = conv_block(32, 3, 2, x, drop=True)
        # x = conv_block(48, 3, 2, x, drop=True)

        x = conv_block(16, 5, 1, x, drop=True)
        x = MaxPooling2D()(x)
        x = conv_block(16, 5, 1, x, drop=True)
        x = MaxPooling2D()(x)
        x = conv_block(32, 3, 1, x, drop=True)
        x = MaxPooling2D()(x)
        x = conv_block(48, 3, 1, x, drop=True)
        x = MaxPooling2D()(x)

        x1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
        x2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        x3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        x = Concatenate()([x1, x2, x3])
        x = Dropout(drop_rate)(x)
        ####

        fe = Model(inp, x)

    inputs = []
    outputs = []
    input_components_names = dataset.indexes2components_names(input_components)
    output_components_names = dataset.indexes2components_names(
        output_components)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    y = fe(inp)

    y = dense_block(150, y, drop=True)
    y = dense_block(75, y, drop=True)

    if 'speed' in input_components_names:
        inp = Input((1,))
        inputs.append(inp)
        y = Concatenate()([y, inp])

    y = dense_block(50, y, drop=False)

    if 'direction' in output_components_names:
        z = Dense(1, use_bias=use_bias,
                  activation=last_act, name="steering")(y)
        outputs.append(z)
        y = Concatenate()([y, z])

    if 'throttle' in output_components_names:
        y = dense_block(50, y, drop=False)
        z = Dense(1, use_bias=use_bias,
                  activation=last_act, name="throttle")(y)
        outputs.append(z)

    model = Model(inputs, outputs)
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
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
    # new_model = flatten_model('test_model\\models\\lightv6_mix.h5')
    new_model = create_light_CNN((120, 160, 3), 5)
