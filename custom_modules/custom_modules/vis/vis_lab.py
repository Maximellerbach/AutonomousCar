import cv2
import numpy as np

from .. datasets.dataset_json import Dataset


def lane(img, lane, color=(0, 0, 255), show=True, name="img", waitkey=None):
    def rescale(lane, shape):
        h, w, ch = shape
        rescale_array = np.array((w, h, w, h)) / 2
        lane = np.array(lane)
        return np.array((lane + 1) * rescale_array, dtype=np.int32)

    lane = rescale(lane, img.shape)
    p1 = tuple(lane[:2])
    p2 = tuple(lane[2:])

    cv2.line(img, p1, p2, color)

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    return img


def direction(img, direction, color=(0, 0, 255), show=True, name="img", waitkey=None):
    h, w, ch = img.shape
    p1 = (w // 2, h)
    p2 = (int(w // 2 + direction * 40), h - 40)

    cv2.line(img, p1, p2, color)

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    return img


def throttle(img, throttle, color=(0, 0, 255), show=True, name="img", waitkey=None, offset=5):
    h, w, ch = img.shape
    p1 = (offset, h)
    p2 = (offset, h - int(throttle * 30))

    cv2.line(img, p1, p2, color, thickness=2)

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    return img


def vis_all(Dataset: Dataset, input_components, img, output_dict, show=True, waitkey=1):
    for output_name in output_dict:
        component = Dataset.get_component(output_name)
        img = component.vis_func(img, output_dict[component.name], show=False)

    if show:
        cv2.imshow("img", img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    return img


def vis_all_compare(Dataset: Dataset, input_components, img, gt_dict, output_dict, show=True, waitkey=1):
    for output_name in output_dict:
        component = Dataset.get_component(output_name)
        img = component.vis_func(img, output_dict[output_name], color=(0, 0, 255), show=False)

        img = component.vis_func(img, gt_dict[component.name], color=(255, 0, 0), show=False)

    if show:
        cv2.imshow("compare", img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    return img
