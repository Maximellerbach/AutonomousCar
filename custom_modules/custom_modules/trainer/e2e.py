import collections
import time

import numpy as np
import tensorflow
import tensorflow_model_optimization as tfmot
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from .. import architectures, imaugm
from ..datagenerator import image_generator
from ..vis import plot

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


class End2EndTrainer():
    """end to end model trainer class."""

    def __init__(self, load_path, save_path, dataset, dospath='', dosdir=True, proportion=0.15, sequence=False,
                 smoothing=0, label_rdm=0, input_components=[], output_components=[0]):
        """Init the trainer parameters.

        Args:
            load_path (str): the path of the model you want to load.
            save_path (str): the path of the future model you want to save.
            dospath (str, optional): path to the directory where the images are stored. Defaults to ''.
            dosdir (bool, optional): is the dospath a directory of directory ? Defaults to True.
            proportion (float, optional): proportion of augmented images for every augmentation functions. Defaults to 0.15.
            sequence (bool, optional): wether to process the image in sequence, not supported for training yet. Defaults to False.
            smoothing (int, optional): value for label smoothing. Defaults to 0.
            label_rdm (int, optional): value for label randomization. Defaults to 0.
        """
        self.Dataset = dataset
        self.load_path = load_path
        self.save_path = save_path
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

    def build_classifier(self, model_arch, load=False, use_bias=True, prune=0, drop_rate=0.15, regularizer=(0.0, 0.0)):
        """Load a model using a model architectures from architectures.py."""
        if load:
            self.model = architectures.safe_load_model(self.load_path, custom_objects={
                "dir_loss": architectures.dir_loss})
            print('loaded model')

        else:
            self.model = model_arch(
                self.Dataset, (120, 160, 3),
                drop_rate=drop_rate, regularizer=regularizer,
                prev_act="relu", last_act="tanh",
                use_bias=use_bias,
                input_components=self.input_components,
                output_components=self.output_components).build()

        self.add_pruning(prune)
        return self.model

    def add_pruning(self, prune):
        if prune:
            self.model = architectures.create_pruning_model(self.model, prune)
            self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, loss="mse", optimizer=tensorflow.keras.optimizers.Adam, lr=0.001, metrics=["mse"]):
        self.model.compile(
            loss=loss,
            optimizer=optimizer(lr=lr),
            metrics=metrics)

        assert len(self.model.outputs) == len(self.output_components)
        assert len(self.model.inputs)-1 == len(self.input_components)
        self.model.summary()

    def train(self, flip=True, augm=True,
              use_earlystop=False, use_tensorboard=False, use_plateau_lr=False, verbose=0,
              epochs=5, batch_size=64, seq_batchsize=16, show_distr=False, datagen=False):
        """Train the model loaded as self.model."""
        gdos, valdos, frc, datalen = self.get_gdos(flip=flip, show=show_distr)
        print(gdos.shape, valdos.shape)
        logdir = f'logs\\{time.time()}\\'

        if batch_size > len(valdos):
            val_batch_size = len(valdos)
        else:
            val_batch_size = batch_size
        it_per_epochs = len(gdos)/batch_size

        if self.model is None:
            self.build_classifier()

        if use_earlystop:
            earlystop = EarlyStopping(
                monitor='loss',
                min_delta=0,
                patience=(1000//it_per_epochs)+1,
                verbose=verbose,
                restore_best_weights=True)
            self.callbacks.append(earlystop)

        if use_tensorboard:
            tensorboard = TensorBoard(
                log_dir=logdir,
                update_freq='batch'
            )
            self.callbacks.append(tensorboard)

        if use_plateau_lr:
            plateau_lr = ReduceLROnPlateau(
                monitor='loss',
                patience=(1000//it_per_epochs)+1,
                min_lr=0.0001,
                verbose=verbose
            )
            self.callbacks.append(plateau_lr)

        self.model.fit(
            x=image_generator(
                gdos, self.Dataset,
                datalen, frc, batch_size,
                self.input_components, self.output_components,
                sequence=self.sequence,
                seq_batchsize=seq_batchsize,
                augm=augm, flip=flip,
                proportion=self.proportion,
                use_tensorboard=use_tensorboard, logdir=logdir),
            steps_per_epoch=datalen//batch_size, epochs=epochs,
            validation_data=image_generator(
                valdos, self.Dataset,
                datalen, frc, val_batch_size,
                self.input_components, self.output_components,
                sequence=self.sequence,
                seq_batchsize=seq_batchsize,
                augm=augm, flip=flip,
                proportion=self.proportion,
                use_tensorboard=use_tensorboard, logdir=logdir),
            validation_steps=datalen//20//val_batch_size,
            callbacks=self.callbacks, max_queue_size=8, workers=4,
            verbose=verbose
        )

        self.model.save(self.save_path)

    def get_gdos(self, flip=True, show=False):
        """Get list of paths in self.dospath.

        Args:
            flip (bool, optional): wether to flip images, used to calculate the total number of images. Defaults to True.

        Returns:
            tuple: (train_paths, test_paths, weighted_distribution, total number of images)
        """
        st = time.time()
        if self.dosdir:
            # even if self.sequence is set to False, load data in sequence
            gdos = self.Dataset.load_dataset_sorted(self.dospath)

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
        et = time.time()
        elapsed_time = et-st
        print(f"fetched dataset {self.dospath} in {elapsed_time} seconds")
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
                    if component.flip:
                        labels = [lab, component.flip_item(lab)]
                    else:
                        labels = [lab, lab]
                else:
                    labels = [lab]

                for label in labels:
                    rounded = imaugm.round_st(label, component.weight_acc)
                    Y[it].append(rounded)

        frcs = []
        for it, index in enumerate(self.output_components+self.input_components):
            component = self.Dataset.get_component(index)
            if component.iterable:
                frcs.append([1]*len(component.default_flat))

            else:
                Y_component = Y[it]

                d = collections.Counter(Y_component)
                unique = np.unique(Y_component)
                frc = class_weight.compute_class_weight(
                    'balanced', unique, Y_component)
                frcs.append(dict(zip(unique, frc)))

                if show:
                    plot.plot_bars(d, component.weight_acc,
                                   title=component.name)

        return frcs
