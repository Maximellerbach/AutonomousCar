import time

import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate,
                                     Conv2D, SeparableConv2D, Dense,
                                     Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.losses import mae, mse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1_l2


def dir_loss(y_true, y_pred):
    '''Loss function for the models.'''
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


def create_pruning_model(model, sparsity=0.5, clone_function=apply_pruning_to_dense_and_conv):
    import tensorflow_model_optimization as tfmot
    return tensorflow.keras.models.clone_model(
        model,
        clone_function=clone_function)


def get_fe(model):
    start_index = 0
    end_index = -1
    for it, layer in enumerate(model.layers):
        if layer.name == 'start_fe':
            start_index = it
        if layer.name == 'end_fe':
            end_index = it

    return Model(model.layers[start_index].input, model.layers[end_index].output)


def get_flat_fe(model):
    fe = get_fe(model)

    if len(fe.layers[-1].output_shape) == 2:
        return fe
    else:
        inp = Input(shape=(120, 160, 3))
        x = fe(inp)
        x = Flatten()(x)
        return Model(inp, x)


def safe_load_model(*args, **kwargs):
    try:
        return load_model(*args, **kwargs)
    except ValueError:
        import tensorflow_model_optimization as tfmot
        with tfmot.sparsity.keras.prune_scope():
            return load_model(*args, **kwargs)


def get_model_output_names(model):
    return [node.op.name.split('/')[0] for node in model.outputs]


def prediction2dict(predictions, model_output_names):
    predictions_list = [[]]*len(predictions[0])
    for prediction in predictions:
        for pred_number, pred in enumerate(prediction):
            predictions_list[pred_number].append(pred)

    output_dicts = [{output_name: [] for output_name in model_output_names}
                    for _ in range(len(predictions_list))]
    for prediction, output_dict in zip(predictions_list, output_dicts):
        for output_value, output_name in zip(prediction, output_dict):
            output_dict[output_name] = K.eval(output_value)
    return output_dicts


def predict_decorator(func, model_output_names):
    def wrapped_f(*args, **kwargs):
        st = time.time()
        predictions = func(*args, **kwargs)
        output_dicts = prediction2dict(predictions, model_output_names)
        et = time.time()
        return output_dicts, et-st
    return wrapped_f


def apply_predict_decorator(model):
    model.predict = predict_decorator(
        model, get_model_output_names(model))


def predict(model, x):
    prediction = model(x, training=False)
    return prediction


class light_linear_CRNN():
    def __init__(self, dataset, img_shape,
                 prev_act='relu', last_act='linear', padding='same',
                 drop_rate=0.1, use_bias=False, regularizer=(0, 0),
                 input_components=[], output_components=[]):

        self.dataset = dataset
        self.img_shape = img_shape
        self.prev_act = prev_act
        self.last_act = last_act
        self.padding = padding
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.input_components = input_components
        self.output_components = output_components

    def rnn_conv_block(self, n_filter, kernel_size, strides, x,
                       conv_type=Conv2D, drop=True,
                       flatten=False, batchnorm=True, **kwargs):

        x = TD(conv_type(
            n_filter,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            bias_regularizer=l1_l2(self.regularizer[0], self.regularizer[1])),
            **kwargs
        )(x)

        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(self.prev_act))(x)
        if drop:
            x = TD(Dropout(self.drop_rate))(x)
        if flatten:
            x = TD(Flatten())(x)
        return x

    def rnn_dense_block(self, n_neurones, x,
                        drop=True, batchnorm=True):
        x = TD(Dense(n_neurones, use_bias=self.use_bias))(x)
        if batchnorm:
            x = TD(BatchNormalization())(x)
        x = TD(Activation(self.prev_act))(x)
        if drop:
            x = TD(Dropout(self.drop_rate))(x)
        return x

    def build(self):
        inputs = []
        outputs = []

        input_components_names = self.dataset.indexes2components_names(
            self.input_components)
        output_components_names = self.dataset.indexes2components_names(
            self.output_components)

        inp = Input(shape=self.img_shape)
        inputs.append(inp)

        x = self.rnn_conv_block(12, 5, 2, inp, drop=True, name='start_fe')
        x = self.rnn_conv_block(16, 5, 2, x, drop=True)
        x = self.rnn_conv_block(32, 3, 2, x, drop=True)
        x = self.rnn_conv_block(48, 3, 2, x, drop=True, name='end_fe')

        y1 = self.rnn_conv_block(64, (8, 10), (8, 10),
                                 x, flatten=True, drop=False)
        y2 = self.rnn_conv_block(
            24, (8, 1), (8, 1), x, flatten=True, drop=False)
        y3 = self.rnn_conv_block(24, (1, 10), (1, 10),
                                 x, flatten=True, drop=False)
        y = Concatenate()([y1, y2, y3])
        y = TD(Dropout(self.drop_rate))(y)

        y = self.rnn_dense_block(150, y, batchnorm=False)
        y = self.rnn_dense_block(75, y, batchnorm=False)

        if 'speed' in input_components_names:
            inp = Input((self.img_shape[0], 1))
            inputs.append(inp)
            y = Concatenate()([y, inp])

        y = self.rnn_dense_block(50, y, batchnorm=False, drop=False)

        if 'direction' in input_components_names:
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation=self.last_act,
                      name='direction')(y)  # kernel_regularizer=l2(0.0005)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'throttle' in output_components_names:
            th = Dense(1,
                       use_bias=self.use_bias,
                       activation='sigmoid',
                       name='throttle')(y)
            outputs.append(th)
            y = Concatenate()([y, th])

        return Model(inputs, outputs)


class light_linear_CNN():
    def __init__(self, dataset, img_shape,
                 prev_act='relu', last_act='linear', padding='same',
                 drop_rate=0.1, use_bias=False, regularizer=(0, 0),
                 input_components=[], output_components=[]):

        self.dataset = dataset
        self.img_shape = img_shape
        self.prev_act = prev_act
        self.last_act = last_act
        self.padding = padding
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.input_components = input_components
        self.output_components = output_components

    def conv_block(self, n_filter, kernel_size, strides, x,
                   conv_type=Conv2D, drop=True,
                   flatten=False, batchnorm=True, maxpool=False, **kwargs):

        x = conv_type(
            n_filter, kernel_size=kernel_size,
            strides=strides, use_bias=self.use_bias, padding=self.padding,
            kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            bias_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            **kwargs
        )(x)

        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.prev_act)(x)
        if drop:
            x = Dropout(self.drop_rate)(x)
        if maxpool:
            x = MaxPooling2D()(x)
        if flatten:
            x = Flatten()(x)
        return x

    def dense_block(self, n_neurones, x,
                    drop=True, batchnorm=True, activation=None, **kwargs):
        x = Dense(
            n_neurones,
            use_bias=self.use_bias,
            kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            bias_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            **kwargs
        )(x)

        if batchnorm:
            x = BatchNormalization()(x)
        if activation is None:
            x = Activation(self.prev_act)(x)
        else:
            x = Activation(activation)(x)
        if drop:
            x = Dropout(self.drop_rate)(x)
        return x

    def build(self):
        inputs = []
        outputs = []

        input_components_names = self.dataset.indexes2components_names(
            self.input_components)
        output_components_names = self.dataset.indexes2components_names(
            self.output_components)

        inp = Input(shape=self.img_shape)
        inputs.append(inp)

        x = BatchNormalization(name='start_fe')(inp)
        x = self.conv_block(6, 3, 2, x, drop=True)
        x = self.conv_block(12, 3, 2, x, drop=True)
        x = self.conv_block(24, 3, 2, x, drop=True)
        x = self.conv_block(24, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        # useless layer, just here to have a "end_fe" layer
        x = Activation('linear', name='end_fe')(x)

        y = Flatten()(x)
        y = Dropout(self.drop_rate)(y)

        y = self.dense_block(75, y, drop=True)
        y = self.dense_block(50, y, drop=True)

        if 'speed' in input_components_names:
            inp = Input((1,),
                        name='speed')
            inputs.append(inp)
            y = Concatenate()([y, inp])
            y = self.dense_block(50, y, drop=False)

        # y = self.dense_block(50, y, drop=False)

        if 'left_lane' in output_components_names:
            z = Dense(4,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='left_lane')(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'right_lane' in output_components_names:
            z = Dense(4,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='right_lane')(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'cte' in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='cte')(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'direction' in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = self.dense_block(25, y, drop=False, activation='softmax')
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation=self.last_act,
                      name='direction')(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'throttle' in output_components_names:
            y = self.dense_block(50, y, drop=False)
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation=self.last_act,
                      name='throttle')(y)
            outputs.append(z)

        return Model(inputs, outputs)


class heavy_linear_CNN():
    def __init__(self, dataset, img_shape,
                 prev_act='relu', last_act='linear', padding='same',
                 drop_rate=0.1, use_bias=False, regularizer=(0, 0),
                 input_components=[], output_components=[]):

        self.dataset = dataset
        self.img_shape = img_shape
        self.prev_act = prev_act
        self.last_act = last_act
        self.padding = padding
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.input_components = input_components
        self.output_components = output_components

    def conv_block(self, n_filter, kernel_size, strides, x,
                   conv_type=Conv2D, drop=True,
                   flatten=False, batchnorm=True, maxpool=False, **kwargs):

        x = conv_type(
            n_filter, kernel_size=kernel_size,
            strides=strides, use_bias=self.use_bias, padding=self.padding,
            kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            bias_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            **kwargs
        )(x)

        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.prev_act)(x)
        if drop:
            x = Dropout(self.drop_rate)(x)
        if maxpool:
            x = MaxPooling2D()(x)
        if flatten:
            x = Flatten()(x)
        return x

    def dense_block(self, n_neurones, x,
                    drop=True, batchnorm=True, activation=None, **kwargs):
        x = Dense(
            n_neurones,
            use_bias=self.use_bias,
            kernel_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            bias_regularizer=l1_l2(self.regularizer[0], self.regularizer[1]),
            **kwargs
        )(x)

        if batchnorm:
            x = BatchNormalization()(x)
        if activation is None:
            x = Activation(self.prev_act)(x)
        else:
            x = Activation(activation)(x)
        if drop:
            x = Dropout(self.drop_rate)(x)
        return x

    def build(self):
        inputs = []
        outputs = []

        input_components_names = self.dataset.indexes2components_names(
            self.input_components)
        output_components_names = self.dataset.indexes2components_names(
            self.output_components)

        inp = Input(shape=self.img_shape)
        inputs.append(inp)

        x = BatchNormalization(name='start_fe')(inp)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        # useless layer, just here to have a "end_fe" layer
        x = Activation('linear', name='end_fe')(x)

        y1 = self.conv_block(32, (8, 10), (8, 10), x, flatten=True, drop=False)
        y2 = self.conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        y3 = self.conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        y = Concatenate()([y1, y2, y3])
        y = Dropout(self.drop_rate)(y)
        # y = self.conv_block(48, 3, 3, x, flatten=True, drop=True)

        y = self.dense_block(150, y, drop=True)
        y = self.dense_block(75, y, drop=True)

        if 'speed' in input_components_names:
            inp = Input((1,),
                        name='speed')
            inputs.append(inp)
            y = Concatenate()([y, inp])
            y = self.dense_block(50, y, drop=False)

        y = self.dense_block(50, y, drop=False)

        if 'left_lane' in output_components_names:
            z = Dense(4,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='left_lane')(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'right_lane' in output_components_names:
            z = Dense(4,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='right_lane')(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'cte' in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation='tanh',
                      name='cte')(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'direction' in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = self.dense_block(25, y, drop=False, activation='softmax')
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation=self.last_act,
                      name='direction')(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if 'throttle' in output_components_names:
            y = self.dense_block(50, y, drop=False)
            z = Dense(1,
                      use_bias=self.use_bias,
                      activation=self.last_act,
                      name='throttle')(y)
            outputs.append(z)

        return Model(inputs, outputs)
