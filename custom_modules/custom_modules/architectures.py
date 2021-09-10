import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.losses import mae, mse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1_l2

try:
    import tensorflow_model_optimization as tfmot
except Exception:
    print("tfmot couldn't be imported")


def get_flops(load_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = load_model(load_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts)

            return flops.total_float_ops


def dir_loss(y_true, y_pred):
    """Loss function for the models."""
    return mae(y_true, y_pred) / 2 + mse(y_true, y_pred)


def linear2dir(linear, dir_range=(3, 11), to_int=True):
    delta_range = dir_range[1] - dir_range[0]
    direction = (((linear + 1) / 2) * delta_range) + dir_range[0]
    if to_int:
        direction = round(direction)
    return direction


def cat2linear(ny):
    averages = []
    for n in ny:
        average = 0
        coef = [-1, -0.5, 0, 0.5, 1]

        for it, nyx in enumerate(n):
            average += nyx * coef[it]
        averages.append(average)
    return averages


def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def apply_pruning_to_conv(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def apply_pruning_to_dense_and_conv(layer):
    if (
        isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.SeparableConv2D)
        or isinstance(layer, tf.keras.layers.Dense)
    ):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def create_pruning_model(model, sparsity=0.5, clone_function=apply_pruning_to_dense_and_conv):
    return tf.keras.models.clone_model(model, clone_function=clone_function)


def get_fe(model):
    start_index = 0
    end_index = -1
    for it, layer in enumerate(model.layers):
        if layer.name == "start_fe":
            start_index = it
        if layer.name == "end_fe":
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
        with tfmot.sparsity.keras.prune_scope():
            return load_model(*args, **kwargs)


def get_model_output_names(model):
    return [node.op.name.split("/")[0] for node in model.outputs]


def get_model_input_names(model):
    return [node.op.name.split("/")[0] for node in model.inputs]


def prediction2dict(predictions, model_output_names):
    predictions_list = [[]] * len(predictions[0])
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
        predictions = func(*args, **kwargs, training=False)
        output_dicts = prediction2dict(predictions, model_output_names)
        et = time.time()
        return output_dicts, et - st

    return wrapped_f


def apply_predict_decorator(model):
    model.predict = predict_decorator(model, get_model_output_names(model))


def predict(model, x):
    prediction = model(x, training=False)
    return prediction


def keras_model_to_tflite(in_filename, out_filename):
    model = tf.keras.models.load_model(in_filename)
    keras_to_tflite(model, out_filename)


def keras_to_tflite(model, out_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(out_filename, "wb").write(tflite_model)


class TFLite:
    def __init__(self, model_path, output_names=[]):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.output_names = output_names

    def get_input_shape(self):
        return [inp["shape"] for inp in self.input_details]

    def predict(self, input_data):
        st = time.time()

        for i, inp in enumerate(input_data):
            self.interpreter.set_tensor(self.input_details[i]["index"], inp)
        self.interpreter.invoke()

        output_dict = {}
        if self.output_names != []:
            for tensor, name in zip(self.output_details, self.output_names):
                output_dict[name] = self.interpreter.get_tensor(tensor["index"])[
                    0][0]
        else:
            for tensor in self.output_details:
                output_dict[tensor["name"]] = self.interpreter.get_tensor(tensor["index"])[
                    0][0]

        elapsed_time = time.time() - st
        return output_dict, elapsed_time


class light_linear_CRNN:
    def __init__(
        self,
        dataset,
        img_shape,
        prev_act="relu",
        last_act="linear",
        padding="same",
        drop_rate=0.1,
        use_bias=False,
        regularizer=(0, 0),
        input_components=[],
        output_components=[],
    ):

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

    def rnn_conv_block(
        self, n_filter, kernel_size, strides, x, conv_type=Conv2D, drop=True, flatten=False, batchnorm=True, **kwargs
    ):

        x = TD(
            conv_type(
                n_filter,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=self.use_bias,
                padding=self.padding,
                kernel_regularizer=l1_l2(
                    self.regularizer[0], self.regularizer[1]),
                bias_regularizer=l1_l2(
                    self.regularizer[0], self.regularizer[1]),
            ),
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

    def rnn_dense_block(self, n_neurones, x, drop=True, batchnorm=True):
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

        x = self.rnn_conv_block(12, 5, 2, inp, drop=True, name="start_fe")
        x = self.rnn_conv_block(16, 5, 2, x, drop=True)
        x = self.rnn_conv_block(32, 3, 2, x, drop=True)
        x = self.rnn_conv_block(48, 3, 2, x, drop=True, name="end_fe")

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

        if "speed" in input_components_names:
            inp = Input((self.img_shape[0], 1))
            inputs.append(inp)
            y = Concatenate()([y, inp])

        y = self.rnn_dense_block(50, y, batchnorm=False, drop=False)

        if "direction" in input_components_names:
            z = Dense(1, use_bias=self.use_bias, activation=self.last_act, name="direction")(
                y
            )  # kernel_regularizer=l2(0.0005)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "throttle" in output_components_names:
            th = Dense(1, use_bias=self.use_bias,
                       activation="sigmoid", name="throttle")(y)
            outputs.append(th)
            y = Concatenate()([y, th])

        return Model(inputs, outputs)


class light_linear_CNN:
    def __init__(
        self,
        dataset,
        img_shape,
        prev_act="relu",
        last_act="linear",
        drop_rate=0.1,
        use_bias=False,
        regularizer=(0, 0),
        input_components=[],
        output_components=[],
    ):

        self.dataset = dataset
        self.img_shape = img_shape
        self.prev_act = prev_act
        self.last_act = last_act
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.input_components = input_components
        self.output_components = output_components

    def conv_block(
        self,
        n_filter,
        kernel_size,
        strides,
        x,
        drop=True,
        conv_type=Conv2D,
        flatten=False,
        batchnorm=True,
        maxpool=False,
        **kwargs
    ):

        x = conv_type(
            n_filter,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=self.use_bias,
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

    def dense_block(self, n_neurones, x, drop=True, batchnorm=False, activation=None, **kwargs):
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

        inp = Input(shape=self.img_shape, name="image")
        inputs.append(inp)

        x = BatchNormalization(name="start_fe")(inp)
        x = self.conv_block(12, 5, 2, x, drop=True, conv_type=SeparableConv2D)
        x = self.conv_block(24, 5, 2, x, drop=True, conv_type=SeparableConv2D)
        x = self.conv_block(32, 3, 2, x, drop=True, conv_type=SeparableConv2D)
        x = self.conv_block(32, 3, 1, x, drop=True, conv_type=SeparableConv2D)
        x = self.conv_block(8, 3, 1, x, drop=True, conv_type=SeparableConv2D)
        # useless layer, just here to have a "end_fe" layer
        x = Activation("linear", name="end_fe")(x)

        y1 = self.conv_block(64, (9, 14), 1, x, drop=True,
                             conv_type=SeparableConv2D)
        y2 = MaxPooling2D()(x)
        y = Concatenate()([Flatten()(y1), Flatten()(y2)])

        y = Dropout(self.drop_rate)(y)

        y = self.dense_block(150, y, drop=True)
        y = self.dense_block(75, y, drop=True)

        if "speed" in input_components_names:
            inp = Input((1,), name="speed")
            inputs.append(inp)
            y = Concatenate()([y, inp])
            y = self.dense_block(50, y, drop=False)

        # y = self.dense_block(50, y, drop=False)

        if "left_lane" in output_components_names:
            z = Dense(4, use_bias=self.use_bias,
                      activation="tanh", name="left_lane")(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "right_lane" in output_components_names:
            z = Dense(4, use_bias=self.use_bias,
                      activation="tanh", name="right_lane")(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "cte" in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = Dense(1, use_bias=self.use_bias,
                      activation="tanh", name="cte")(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "direction" in output_components_names:
            z = self.dense_block(50, y, drop=False)
            z = self.dense_block(25, z, drop=False)
            z = self.dense_block(9, z, drop=False, activation="softmax")
            z = Dense(1, use_bias=self.use_bias,
                      activation=self.last_act, name="direction")(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "throttle" in output_components_names:
            y = self.dense_block(50, y, drop=False)
            z = Dense(1, use_bias=self.use_bias,
                      activation=self.last_act, name="throttle")(y)
            outputs.append(z)

        return Model(inputs, outputs)


class heavy_linear_CNN:
    def __init__(
        self,
        dataset,
        img_shape,
        prev_act="relu",
        last_act="linear",
        drop_rate=0.1,
        use_bias=False,
        regularizer=(0, 0),
        input_components=[],
        output_components=[],
    ):

        self.dataset = dataset
        self.img_shape = img_shape
        self.prev_act = prev_act
        self.last_act = last_act
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.input_components = input_components
        self.output_components = output_components

    def conv_block(
        self,
        n_filter,
        kernel_size,
        strides,
        x,
        conv_type=Conv2D,
        drop=True,
        flatten=False,
        batchnorm=True,
        maxpool=False,
        **kwargs
    ):

        x = conv_type(
            n_filter,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=self.use_bias,
            padding=self.padding,
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

    def dense_block(self, n_neurones, x, drop=True, batchnorm=True, activation=None, **kwargs):
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

        x = BatchNormalization(name="start_fe")(inp)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        x = self.conv_block(32, 3, 2, x, drop=True)
        # useless layer, just here to have a "end_fe" layer
        x = Activation("linear", name="end_fe")(x)

        y1 = self.conv_block(32, (8, 10), (8, 10), x, flatten=True, drop=False)
        y2 = self.conv_block(24, (8, 1), (8, 1), x, flatten=True, drop=False)
        y3 = self.conv_block(24, (1, 10), (1, 10), x, flatten=True, drop=False)
        y = Concatenate()([y1, y2, y3])
        y = Dropout(self.drop_rate)(y)
        # y = self.conv_block(48, 3, 3, x, flatten=True, drop=True)

        y = self.dense_block(150, y, drop=True)
        y = self.dense_block(75, y, drop=True)

        if "speed" in input_components_names:
            inp = Input((1,), name="speed")
            inputs.append(inp)
            y = Concatenate()([y, inp])
            y = self.dense_block(50, y, drop=False)

        y = self.dense_block(50, y, drop=False)

        if "left_lane" in output_components_names:
            z = Dense(4, use_bias=self.use_bias,
                      activation="tanh", name="left_lane")(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "right_lane" in output_components_names:
            z = Dense(4, use_bias=self.use_bias,
                      activation="tanh", name="right_lane")(y)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "cte" in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = Dense(1, use_bias=self.use_bias,
                      activation="tanh", name="cte")(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "direction" in output_components_names:
            z = self.dense_block(25, y, drop=False)
            z = self.dense_block(9, y, drop=False, activation="softmax")
            z = Dense(1, use_bias=self.use_bias,
                      activation=self.last_act, name="direction")(z)
            outputs.append(z)
            y = Concatenate()([y, z])

        if "throttle" in output_components_names:
            y = self.dense_block(50, y, drop=False)
            z = Dense(1, use_bias=self.use_bias,
                      activation=self.last_act, name="throttle")(y)
            outputs.append(z)

        return Model(inputs, outputs)
