from custom_modules.datagenerator import image_generator
from custom_modules.datasets import dataset_json
from custom_modules.vis import plot
from custom_modules import trainer

if __name__ == "__main__":
    Dataset = dataset_json.Dataset(['direction', 'time'])
    direction_comp = Dataset.get_component('direction')
    direction_comp.offset = -7
    direction_comp.scale = 1/4

    # set input and output components (indexes)
    input_components = []
    output_components = [0]

    e2e_trainer = trainer.End2EndTrainer(
        name='test_model\\models\\linear_trackmania2.h5',
        dataset=Dataset,
        dospath='C:\\Users\\maxim\\random_data\\json_dataset\\', dosdir=True,
        proportion=0.2, sequence=False,
        smoothing=0.0, label_rdm=0.0,
        input_components=input_components,
        output_components=output_components)

    e2e_trainer.build_classifier(
        load=False,
        drop_rate=0.05, prune=0.0,
        regularizer=(0.0, 0.0005),
        lr=0.001)

    e2e_trainer.train(
        flip=True,
        augm=True,
        use_earlystop=True,
        use_tensorboard=True,
        epochs=7,
        batch_size=48,
        show_distr=False)
