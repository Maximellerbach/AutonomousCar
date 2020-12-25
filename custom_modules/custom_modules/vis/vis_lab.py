import cv2
import numpy as np


def lane(img, lane, show=True, name="img", waitkey=None):
    def rescale(lane, shape):
        h, w, ch = shape
        rescale_array = np.array((w, h, w, h))/2
        lane = np.array(lane)
        return np.array((lane+1)*rescale_array, dtype=np.int32)

    lane = rescale(lane, img.shape)
    p1 = tuple(lane[:2])
    p2 = tuple(lane[2:])

    cv2.line(img, p1, p2, (255, 0, 0))

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    else:
        return img


def direction(img, direction, show=True, name="img", waitkey=None):
    h, w, ch = img.shape
    p1 = (w//2, h)
    p2 = (int(w//2 + direction*40), h-40)

    cv2.line(img, p1, p2, (0, 0, 255))

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    else:
        return img


def throttle(img, throttle, show=True, name="img", waitkey=None, offset=5):
    h, w, ch = img.shape
    p1 = (offset, h)
    p2 = (offset, h-int(throttle*30))

    cv2.line(img, p1, p2, (0, 0, 255), thickness=2)

    if show:
        cv2.imshow(name, img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
    else:
        return img


def vis_all(Dataset, input_components, img, output_dicts, waitkey=1):
    for output_dict in output_dicts:
        for output_name in output_dict:
            component = Dataset.get_component(output_name)
            img = component.vis_func(
                img, output_dict[output_name], show=False)

        cv2.imshow('img', img)
        if waitkey is not None:
            cv2.waitKey(waitkey)
