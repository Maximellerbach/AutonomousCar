import collections
import time

import numpy as np
import tensorflow
import tensorflow_model_optimization as tfmot
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

# import pred_function
from custom_modules import architectures, autolib
from custom_modules.datagenerator import image_generator
from custom_modules.datasets import dataset_json
from custom_modules.vis import plot

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


class model_trainer():
    """model trainer class."""

    def __init__(self, name, dataset, dospath='', dosdir=True, proportion=0.15, sequence=False,
                 smoothing=0, label_rdm=0, input_components=[], output_components=[0]):
        """Init the trainer parameters.

        Args:
            name (str): the name you want your model to be called
            dospath (str, optional): path to the directory where the images are stored. Defaults to ''.
            dosdir (bool, optional): is the dospath a directory of directory ? Defaults to True.
            proportion (float, optional): proportion of augmented images for every augmentation functions. Defaults to 0.15.
            sequence (bool, optional): wether to process the image in sequence, not supported for training yet. Defaults to False.
            smoothing (int, optional): value for label smoothing. Defaults to 0.
            label_rdm (int, optional): value for label randomization. Defaults to 0.
        """
        # self.Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
        self.Dataset = dataset
        self.name = name
        self.dospath = dospath
        self.dosdir = dosdir
        self.sequence = sequence

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.number_class = 5
        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.input_components = input_components
        self.output_components = output_components

        self.callbacks = []
        self.model = None

    def build_classifier(self, load=False, prune=0, drop_rate=0.15, regularizer=(0.0, 0.0),
                         optimizer=tensorflow.keras.optimizers.Adam, lr=0.001,
                         loss=architectures.dir_loss, metrics=["mse"]):
        """Load a model using a model architectures from architectures.py."""

        if load:
            try:
                self.model = load_model(self.name, custom_objects={
                                        "dir_loss": architectures.dir_loss})
                return self.model

            except ValueError:
                with tfmot.sparsity.keras.prune_scope():
                    self.model = load_model(self.name, custom_objects={
                                            "dir_loss": architectures.dir_loss})
                return self.model

        if self.sequence:
            self.model = architectures.create_light_CRNN(
                self.Dataset, (None, 120, 160, 3),
                drop_rate=drop_rate, regularizer=regularizer,
                prev_act="relu", last_act="linear", padding='same',
                use_bias=True,
                input_components=self.input_components,
                output_components=self.output_components)
        else:
            self.model = architectures.create_light_CNN(
                self.Dataset, (120, 160, 3),
                drop_rate=drop_rate, regularizer=regularizer,
                prev_act="relu", last_act="linear", padding='same',
                use_bias=True,
                input_components=self.input_components,
                output_components=self.output_components)

        if prune:
            self.model = architectures.create_pruning_model(self.model, prune)
            pruning = tfmot.sparsity.keras.UpdatePruningStep()
            self.callbacks.append(pruning)

        self.model.compile(
            loss=loss,
            optimizer=optimizer(lr=lr),
            metrics=metrics)

        assert len(self.model.outputs) == len(self.output_components)
        assert len(self.model.inputs)-1 == len(self.input_components)
        self.model.summary()
        return self.model

    def train(self, flip=True, augm=True, use_earlystop=False, use_tensorboard=False,
              epochs=5, batch_size=64, seq_batchsize=16, show_distr=False):
        """Train the model loaded as self.model."""
        gdos, valdos, frc, datalen = self.get_gdos(flip=flip, show=show_distr)
        print(gdos.shape, valdos.shape)
        logdir = f'logs\\{time.time()}\\'

        if self.model is None:
            self.build_classifier()

        if use_earlystop:
            earlystop = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=3,
                verbose=0,
                restore_best_weights=True)
            self.callbacks.append(earlystop)

        if use_tensorboard:
            tensorboard = TensorBoard(
                log_dir=logdir,
                update_freq='batch')
            self.callbacks.append(tensorboard)

        self.model.fit(
            x=image_generator(
                gdos, self.Dataset,
                self.input_components, self.output_components,
                datalen, frc, batch_size,
                sequence=self.sequence,
                seq_batchsize=seq_batchsize,
                augm=augm, flip=flip,
                use_tensorboard=use_tensorboard, logdir=logdir),
            steps_per_epoch=datalen//batch_size, epochs=epochs,
            validation_data=image_generator(
                valdos, self.Dataset,
                self.input_components, self.output_components,
                datalen, frc, batch_size,
                sequence=self.sequence,
                seq_batchsize=seq_batchsize,
                augm=augm, flip=flip,
                use_tensorboard=use_tensorboard, logdir=logdir),
            validation_steps=datalen//20//batch_size,
            callbacks=self.callbacks, max_queue_size=4, workers=4)

        self.model.save(self.name)

    def get_gdos(self, flip=True, show=False):
        """Get list of paths in self.dospath.

        Args:
            flip (bool, optional): wether to flip images, used to calculate the total number of images. Defaults to True.

        Returns:
            tuple: (train_paths, test_paths, weighted_distribution, total number of images)
        """
        if self.dosdir:
            # even if self.sequence is set to False, load data in sequence
            gdos = self.Dataset.load_dataset_sequence(self.dospath)
            gdos = np.concatenate([i for i in gdos])

        else:
            gdos = self.Dataset.load_dos_sorted(self.dospath)
            gdos = self.Dataset.split_sorted_paths(gdos)

        if self.sequence:
            datalen = 0
            for s in gdos:
                datalen += len(s)

            np.random.shuffle(gdos)
            traindos, valdos = np.split(gdos, [len(gdos)-len(gdos)//20])

        else:
            gdos = np.concatenate([i for i in gdos])
            datalen = len(gdos)

            np.random.shuffle(gdos)
            traindos, valdos = np.split(gdos, [datalen-datalen//20])

        frc = self.get_frc_lin(gdos, flip=flip, show=show)
        return traindos, valdos, frc, datalen

    def get_frc_lin(self, gdos, flip=True, show=False):
        """Get the frc dict from gdos with linear labels.

        Args:
            gdos (list): list of paths, could also be a list of paths sequence
            flip (bool, optional): wether to flip images. Defaults to True.
            show (bool, optional): wether to plot the data. Defaults to False.

        Returns:
            dict: dictionnary of label frequency
        """
        def flatten_paths(paths):
            if isinstance(paths[0], list):
                return flatten_paths([p for p in listp for listp in paths])
            elif isinstance(paths[0], str):
                return paths

        Y = [[] for _ in self.output_components+self.input_components]
        for path in flatten_paths(gdos):
            annotation = self.Dataset.load_annotation(path, to_list=False)

            for it, index in enumerate(self.output_components+self.input_components):
                component = self.Dataset.get_component(index)
                lab = annotation[component.name]

                if flip:
                    labels = [lab, lab*component.flip_factor]
                else:
                    labels = [lab]

                for label in labels:
                    Y[it].append(autolib.round_st(label, component.weight_acc))

        frcs = []
        for it, index in enumerate(self.output_components+self.input_components):
            component = self.Dataset.get_component(index)
            Y_component = Y[it]

            d = collections.Counter(Y_component)
            unique = np.unique(Y_component)
            frc = class_weight.compute_class_weight(
                'balanced', unique, Y_component)
            frcs.append(dict(zip(unique, frc)))

            if show:
                plot.plot_bars(d, component.weight_acc)

        return frcs

    def calculate_FLOPS(self):
        """Calculate the number of flops in self.model.

        Returns:
            int: total number of flops
        """
        run_meta = tensorflow.RunMetadata()
        opts = tensorflow.profiler.ProfileOptionBuilder.float_operation()

        # We use the tensorflow.keras session graph in the call to the profiler.
        flops = tensorflow.profiler.profile(graph=K.get_session(
        ).graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


if __name__ == "__main__":
    Dataset = dataset_json.Dataset(['direction', 'time'])
    direction_comp = Dataset.get_component('direction')
    direction_comp.offset = -7
    direction_comp.scale = 1/4

    # set input and output components (indexes)
    input_components = []
    output_components = [0]

    trainer = model_trainer(name='test_model\\models\\linear_trackmania2.h5',
                            dataset=Dataset,
                            dospath='C:\\Users\\maxim\\random_data\\json_dataset\\', dosdir=True,
                            proportion=0.2, sequence=False,
                            smoothing=0.0, label_rdm=0.0,
                            input_components=input_components,
                            output_components=output_components)

    trainer.build_classifier(load=False,
                             drop_rate=0.05, prune=0.0,
                             regularizer=(0.0, 0.0005),
                             lr=0.001)

    trainer.train(flip=True, augm=True,
                  use_earlystop=True, use_tensorboard=True,
                  epochs=7, batch_size=48,
                  show_distr=False)

    # custom_objects={"dir_loss":architectures.dir_loss}
    # trainer.model = load_model(trainer.name, compile=False)
