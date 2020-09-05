from custom_modules.datasets import dataset_json
from custom_modules.trainer import e2e
from custom_modules import pred_function, architectures

if __name__ == "__main__":
    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'left_lane', 'right_lane', 'time'])
    direction_comp = Dataset.get_component('direction')
    direction_comp.offset = -7
    direction_comp.scale = 1/4

    # set input and output components (indexes)
    input_components = []
    output_components = [0, 2, 3, 4]

    e2e_trainer = e2e.End2EndTrainer(
        name='test_model\\models\\test.h5',
        dataset=Dataset,
        dospath='\\\\MSI\\Users\\maxim\\random_data\\1 ironcar driving\\', dosdir=False,
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
        use_tensorboard=True,
        use_plateau_lr=True,
        verbose=1,
        epochs=150,
        batch_size=4,
        show_distr=False)

    model = architectures.safe_load_model('test_model\\models\\test.h5')
    paths = Dataset.load_dos_sorted(
        'C:\\Users\\maxim\\random_data\\json_dataset\\1 ironcar driving\\')
    pred_function.test_paths(Dataset, input_components, model, paths)
