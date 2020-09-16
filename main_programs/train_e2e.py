import os

from custom_modules import architectures, pred_function
from custom_modules.datasets import dataset_json
from custom_modules.trainer import e2e

if __name__ == "__main__":

    # use the onedrive path 
    # as root directory for data paths
    base_path = os.getenv('ONEDRIVE') + "\\random_data"
    # train_path = f'{base_path}\\json_dataset\\2 donkeycar driving\\'
    train_path = 'C:\\Users\\maxim\\recorded_imgs\\'

    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])
    direction_comp = Dataset.get_component('direction')
    # direction_comp.offset = -7
    # direction_comp.scale = 1/4

    # set input and output components (indexes)
    input_components = [1]
    output_components = [0, 2]

    e2e_trainer = e2e.End2EndTrainer(
        name='test_model\\models\\rd_sim.h5',
        dataset=Dataset,
        dospath=train_path, dosdir=True,
        proportion=0.2, sequence=False,
        smoothing=0.0, label_rdm=0.0,
        input_components=input_components,
        output_components=output_components)

    e2e_trainer.build_classifier(
        load=False,
        use_bias=False,
        drop_rate=0.05, prune=0.0,
        regularizer=(0.0, 0.0005),
        loss='mse', lr=0.001, metrics=[])

    e2e_trainer.train(
        flip=True,
        augm=True,
        use_earlystop=False,
        use_tensorboard=False,
        use_plateau_lr=False,
        verbose=True,
        epochs=4,
        batch_size=32,
        show_distr=True)

    model = architectures.safe_load_model('test_model\\models\\rd_sim.h5')
    paths = Dataset.load_dataset(train_path)
    merged_paths = []
    for path in paths:
        merged_paths += path

    pred_function.test_paths(Dataset, input_components,
                             model, merged_paths, waitkey=1)
