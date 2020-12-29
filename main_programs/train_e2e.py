import os

from custom_modules import architectures, pred_function
from custom_modules.datasets import dataset_json
from custom_modules.trainer import e2e

if __name__ == "__main__":

    # use the home path as root directory for data paths
    base_path = os.path.expanduser("~") + "\\random_data"
    train_path = f'{base_path}\\forza2\\'
    dosdir = True

    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle'])

    # Apply some transformations on the components
    speed_comp = Dataset.get_component('speed')
    speed_comp.offset = 0
    speed_comp.scale = 3.6

    # set input and output components (indexes)
    input_components = [1]
    output_components = [0]

    load_path = 'test_model\\models\\forza4.h5'
    save_path = 'test_model\\models\\forza4.h5'

    e2e_trainer = e2e.End2EndTrainer(
        load_path=load_path,
        save_path=save_path,
        dataset=Dataset,
        dospath=train_path, dosdir=dosdir,
        proportion=0.5, sequence=False,
        smoothing=0.0, label_rdm=0.0,
        input_components=input_components,
        output_components=output_components)

    e2e_trainer.build_classifier(
        load=True,
        use_bias=False,
        drop_rate=0.05, prune=0.0,
        regularizer=(0.0, 0.0001))

    e2e_trainer.compile_model(
        loss=architectures.tensorflow.keras.losses.Huber(delta=2),
        lr=0.0005, metrics=[])

    e2e_trainer.train(
        flip=True,
        augm=True,
        use_earlystop=False,
        use_tensorboard=False,
        use_plateau_lr=False,
        verbose=True,
        epochs=3,
        batch_size=64,
        show_distr=True)

    model = architectures.safe_load_model(save_path, compile=False)
    if dosdir:
        paths = Dataset.load_dataset_sorted(train_path, flat=True)
    else:
        paths = Dataset.load_dos_sorted(train_path)

    pred_function.test_compare_paths(Dataset, input_components,
                                     model, paths, waitkey=33)
