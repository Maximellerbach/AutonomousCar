from . import architectures
from .vis import vis_lab


def test_paths(Dataset, input_components, model, paths):
    architectures.apply_predict_decorator(model)

    for path in paths:
        img, annotation = Dataset.load_img_and_annotation(path)

        to_pred = Dataset.make_to_pred([path], input_components)
        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all(Dataset, input_components, model, img, output_dicts)
