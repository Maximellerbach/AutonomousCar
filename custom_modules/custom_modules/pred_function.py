import numpy as np
import cv2

from . import architectures


def test_paths(Dataset, input_components, model, paths):
    architectures.apply_predict_decorator(model)

    for path in paths:
        img, annotation = Dataset.load_img_and_annotation(path)
        
        to_pred = Dataset.make_to_pred([path], input_components)
        output_dicts, elapsed_time = model.predict(to_pred)
        print(elapsed_time)

        for output_dict in output_dicts:
            for output_name in output_dict:
                component = Dataset.get_component(output_name)
                img = component.vis_func(
                    img, output_dict[output_name], show=False)

            cv2.imshow('img', img)
            cv2.waitKey(1)
