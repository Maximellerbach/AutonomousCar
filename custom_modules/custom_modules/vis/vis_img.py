import cv2


def show(img, name="img", waitkey=None):
    cv2.imshow(name, img)
    if waitkey is not None:
        cv2.waitKey(waitkey)
