from . import architectures
from .vis import vis_lab


def test_paths(Dataset, input_components, model, paths, waitkey=0):
    architectures.apply_predict_decorator(model)

    for path in paths:
        img = Dataset.load_img(path)

        to_pred = Dataset.make_to_pred([path], input_components)
        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all(Dataset, input_components, model, img, output_dicts, waitkey=waitkey)
