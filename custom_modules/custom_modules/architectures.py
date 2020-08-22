import tensorflow_model_optimization as tfmot
import tensorflow
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Dense, DepthwiseConv2D, Dropout, Flatten,
                                     MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.losses import mae, mse
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2


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


def apply_pruning_to_dense(layer):
    if isinstance(layer, tensorflow.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def apply_pruning_to_conv(layer):
    if isinstance(layer, tensorflow.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def apply_pruning_to_dense_and_conv(layer):
    if isinstance(layer, tensorflow.keras.layers.Conv2D) or isinstance(layer, tensorflow.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def create_pruning_model(model, sparsity=0.5):
    return tensorflow.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense)


def get_fe(model):
    start_index = 0
    end_index = -1
    for it, layer in enumerate(model.layers):
        if layer.name == 'start_fe':
            start_index = it
        if layer.name == "end_fe":
            end_index = it

    return Model(model.layers[start_index].input, model.layers[end_index].output)


def create_light_CRNN(dataset, img_shape,
                      prev_act="relu", last_act="linear", padding="same",
                      drop_rate=0.1, use_bias=False, regularizer=(0, 0),
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
    outputs = []

    input_components_names = dataset.indexes2components_names(input_components)
    output_components_names = dataset.indexes2components_names(
        output_components)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    x = TD(BatchNormalization(name="start_fe"))(inp)

    x = conv_block(12, 5, 2, x, drop=True)
    x = conv_block(16, 5, 2, x, drop=True)
    x = conv_block(32, 3, 2, x, drop=True)
    x = conv_block(48, 3, 2, x, drop=True, name="end_fe")

    y1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
    y2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
    y3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
    y = Concatenate()([y1, y2, y3])
    y = TD(Dropout(drop_rate))(y)

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

    return Model(inputs, outputs)


def create_light_CNN(dataset, img_shape,
                     prev_act="relu", last_act="linear", padding='same',
                     drop_rate=0.1, use_bias=False, regularizer=(0, 0),
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

    inputs = []
    outputs = []

    input_components_names = dataset.indexes2components_names(input_components)
    output_components_names = dataset.indexes2components_names(
        output_components)

    inp = Input(shape=img_shape)
    inputs.append(inp)
    x = BatchNormalization(name="start_fe")(inp)

    x = conv_block(16, 5, 1, x, drop=True)
    x = MaxPooling2D()(x)
    x = conv_block(24, 3, 1, x, drop=True)
    x = MaxPooling2D()(x)
    x = conv_block(32, 3, 1, x, drop=True)
    x = MaxPooling2D()(x)
    x = conv_block(48, 3, 1, x, drop=True)
    x = MaxPooling2D(name="end_fe")(x)

    y1 = conv_block(64, (8, 10), (8, 10), x, flatten=True, drop=False)
    y2 = conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
    y3 = conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
    y = Concatenate()([y1, y2, y3])
    y = Dropout(drop_rate)(y)

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

    return Model(inputs, outputs)
