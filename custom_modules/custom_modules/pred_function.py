from . import architectures
from .vis import vis_lab


def test_predict_paths(Dataset, input_components, model, paths, waitkey=0, apply_decorator=True):
    if apply_decorator:
        architectures.apply_predict_decorator(model)

    for path in paths:
        img = Dataset.load_img(path)
        to_pred = Dataset.make_to_pred_paths([path], input_components)

        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all(Dataset, input_components, img, output_dicts[0], waitkey=waitkey)


def test_compare_paths(Dataset, input_components, model, paths, waitkey=0, apply_decorator=True):
    architectures.apply_predict_decorator(model)

    for path in paths:
        img, real_annotations = Dataset.load_img_and_annotation(path, to_list=False)
        to_pred = Dataset.make_to_pred_paths([path], input_components)

        output_dicts, elapsed_time = model.predict(to_pred)
        vis_lab.vis_all_compare(Dataset, input_components, img, real_annotations, output_dicts[0], waitkey=waitkey)
