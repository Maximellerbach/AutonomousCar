import os

from custom_modules import architectures, pred_function
from custom_modules.datasets import dataset_json
from custom_modules.trainer import e2e

if __name__ == "__main__":

    # use the home path as root directory for data paths
    base_path = os.path.expanduser("~") + "\\random_data"
    train_path = f'{base_path}\\forza2\\'
    test_path = f'{base_path}\\forza\\'
    dosdir = True

    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle'])

    # Apply some transformations on the components
    # speed_comp = Dataset.get_component('speed')
    # speed_comp.offset = 0
    # speed_comp.scale = 3.6

    # set input and output components (indexes)
    input_components = []
    output_components = [0]

    load_path = 'test_model\\models\\forza4.h5'
    save_path = 'test_model\\models\\forza4.h5'

    e2e_trainer = e2e.End2EndTrainer(
        load_path=load_path,
        save_path=save_path,
        dataset=Dataset,
        dospath=train_path, dosdir=dosdir,
        proportion=0.3, sequence=False,
        smoothing=0.0, label_rdm=0.0,
        input_components=input_components,
        output_components=output_components)

    e2e_trainer.build_classifier(
        architectures.light_linear_CNN,
        load=False,
        use_bias=False,
        drop_rate=0.05, prune=0.0,
        regularizer=(0.0, 0.0))

    e2e_trainer.compile_model(
        loss=architectures.tensorflow.keras.losses.Huber(delta=2),
        lr=0.001, metrics=[])

    e2e_trainer.train(
        flip=True,
        augm=True,
        use_earlystop=False,
        use_tensorboard=False,
        use_plateau_lr=False,
        verbose=True,
        epochs=8,
        batch_size=32,
        show_distr=False)

    print(architectures.get_flops(save_path))
    model = architectures.safe_load_model(save_path, compile=False)

    if dosdir:
        paths = Dataset.load_dataset_sorted(test_path, flat=True)
    else:
        paths = Dataset.load_dos_sorted(test_path)

    pred_function.test_compare_paths(Dataset, input_components,
                                     model, paths, waitkey=1)
