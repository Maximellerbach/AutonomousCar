from . import architectures
from .vis import vis_lab


def test_predict_paths(Dataset, input_components, model, paths, waitkey=0, apply_decorator=True):
    if apply_decorator:
        architectures.apply_predict_decorator(model)

    for path in paths:
        img = Dataset.load_img(path)
        to_pred = Dataset.make_to_pred([path], input_components)

        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all(Dataset, input_components, model,
                        img, output_dicts, waitkey=waitkey)


def test_compare_paths(Dataset, input_components, model, paths, waitkey=0):  # not implemented yet
    architectures.apply_predict_decorator(model)

    for path in paths:
        img, real_annotations = Dataset.load_img_and_annotation(path)
        to_pred = Dataset.make_to_pred([path], input_components)

        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all(Dataset, input_components, model,
                        img, output_dicts, waitkey=waitkey)
