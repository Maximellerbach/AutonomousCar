import cv2
import numpy as np
import math


def camera_dist_mask(color=np.array([1, 1, 1]), shape=(120, 160, 3)):
    mask = np.zeros(shape, dtype=np.float)

    for yi in range(shape[0]):
        plan_dist = shape[0] - yi
        mask[yi, :] = np.array([color * plan_dist] * shape[1])

    max_value = np.max(mask)
    return mask / max_value  # normalize the mask


def compute_mask(fish_eye_x, fish_eye_y, color=np.array([1, 1, 1]), shape=(120, 160, 3)):
    mask = np.zeros(shape, dtype=np.float)
    center = (shape[0] // 2, shape[1] // 2)

    for xi in range(shape[1]):
        for yi in range(shape[0]):
            x_dist = ((xi - center[1]) ** 2) * fish_eye_x
            y_dist = ((yi - center[0]) ** 2) * fish_eye_y

            mask[yi, xi] = (math.sqrt(x_dist + y_dist)) * color

    max_value = np.max(mask)
    return mask / max_value  # normalize the mask


def main():
    image = cv2.imread("C:\\Users\\maxim\\random_data\\donkeycar\\donkeycar\\1570868254.707678.png") / 255
    mask = compute_mask(0.6, 0.8)
    cam_dist_mask = camera_dist_mask()
    combined_mask = 1 - (mask + cam_dist_mask) / 2

    cv2.imshow("mask", combined_mask)
    cv2.imshow("masked", image * combined_mask)
    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
