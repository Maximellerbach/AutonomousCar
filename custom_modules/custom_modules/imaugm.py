import collections
import random

import cv2
import numpy as np
import skimage.exposure as sk

low_y = np.array([31, 60, 60])
up_y = np.array([50, 255, 255])

low_w = np.array([0, 0, 190])
up_w = np.array([255, 10, 255])

dico = [3, 5, 7, 9, 11]
rev = [11, 9, 7, 5, 3]


def round_st(st, acc=0.5):
    n_val = 1/acc
    if isinstance(st, collections.Iterable):
        return [round(st_value*n_val, 0)/n_val for st_value in st]
    else:
        return round(st*n_val, 0)/n_val


def get_weight(Y, frc, is_cat, acc=0.5):
    w = []
    for y in Y:
        r = round_st(y, acc)
        w.append(frc[r])

    return np.array(w)


def cut_img(img, c):
    img = img[c:, :, :]
    return img


def label_smoothing(Y, n, k, random=0):
    smooth_y = []
    if random != 0:
        k = k+np.random.random()*random
    for y in Y:
        sy = [0]*n
        sy[y] = 1-k
        if y == 2:
            sy[y-1] = k/2
            sy[y+1] = k/2

        elif y == 0:
            sy[y+1] = k

        elif y == -1:
            sy[y-1] = k

        elif y == 1:
            sy[y-1] = k*2/3
            sy[y+1] = k*1/3

        elif y == 3:
            sy[y-1] = k*1/3
            sy[y+1] = k*2/3

        smooth_y.append(sy)
    return np.array(smooth_y)


def change_brightness(img, label, value=np.random.randint(15, 45), sign=np.random.choice([True, False])):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if sign:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] = v[v <= lim]+value
    if sign:
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] = v[v >= lim]-value

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, label


def rescut(image, label):
    rdm_cut = int(np.random.uniform(0, 20))
    sign = np.random.choice([True, False])

    if sign:
        img = image[:, rdm_cut:, :]
    else:
        dif = image.shape[1]-rdm_cut
        img = image[:, :dif, :]

    return cv2.resize(img, (160, 120)), label


def add_random_shadow(image, label):
    shape = image.shape
    top_y = shape[1]*np.random.uniform()
    top_x = shape[0]*np.random.uniform()
    bot_x = shape[0]*np.random.uniform()
    bot_y = shape[1]*np.random.uniform()

    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:, :, 1]

    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -
                 (bot_x - top_x)*(Y_m-top_y) >= 0)] = 1

    sign = np.random.choice([True, False])
    if sign:
        random_bright = 1+0.5*np.random.uniform()
    else:
        random_bright = 1-0.5*np.random.uniform()

    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0

    if np.random.randint(2) == 1:
        image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1]*random_bright
    else:
        image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0]*random_bright

    new_image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)
    return new_image, label


def night_effect(img,  label, vmin=150, vmax=230):
    limit = random.uniform(vmin, vmax)
    low_limit = vmin
    int_img = sk.rescale_intensity(img, in_range=(
        low_limit, limit), out_range=(0, 255))

    return int_img, label


def horizontal_flip(Dataset, names2index, flip_indexes, img, labels, index):
    original_labels = np.array(labels)
    labels = np.array(labels)
    for flip_index in flip_indexes:
        component = Dataset.get_component(flip_index)
        if component.is_couple:
            couple_index = names2index[component.couple]
            labels[flip_index][index] = component.flip_item(
                original_labels[couple_index][index])
        else:
            labels[flip_index][index] = component.flip_item(
                original_labels[flip_index][index])

    return cv2.flip(img, 1), labels


def add_rdm_noise(img, label):
    img = img+np.random.uniform(-25, 25, size=img.shape)
    return img, label


def inverse_color(img, label):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    rdm_c = np.random.uniform(0.6, 1.4, 3)
    order = [b*rdm_c[0], g*rdm_c[1], r*rdm_c[2]]
    random.shuffle(order)
    img = cv2.merge(order)
    img = img*(1/max(rdm_c))

    return img, label


def mirror_image(img, label):
    center = img.shape[0]//2
    side = np.random.choice([True, False])

    if side:
        tmp_img = img[center:, :, :]
        flip_tmp = cv2.flip(tmp_img, 1)
        img = np.concatenate((tmp_img, flip_tmp), axis=1)
    else:
        tmp_img = img[:center, :, :]
        flip_tmp = cv2.flip(tmp_img, 1)
        img = np.concatenate((flip_tmp, tmp_img), axis=1)

    return img, label


def generate_functions(X, Y, proportion=0.25):
    functions = (change_brightness, rescut, inverse_color,
                 night_effect, add_random_shadow, add_rdm_noise)
    length_X = len(X)

    X_aug = []
    Y_aug = []
    for f in functions:
        indexes = np.random.choice([True, False], length_X, p=[
                                   proportion, 1-proportion])

        for index in range(length_X):
            if indexes[index]:
                im, annotation = f(X[index], Y[index])
                im = np.array(im, dtype=np.uint8)
                Y_aug.append(annotation)
                X_aug.append(im)

    return X_aug, Y_aug


def generate_functions_replace(X, Y, proportion=0.25,
                               functions=(change_brightness,
                                          rescut,
                                          inverse_color,
                                          night_effect,
                                          add_random_shadow,
                                          add_rdm_noise)):
    X = np.array(X)
    Y = np.array(Y)

    # same function as above, just replace images instead of adding more
    for f in functions:

        indexes = np.random.choice([True, False], len(X), p=[
                                   proportion, 1-proportion])
        for index in range(len(X)):
            if indexes[index]:
                im, annotation = f(X[index], Y[:, index])
                im = np.clip(im, 0, 255)  # force image to be between 0 and 255
                im = np.array(im, dtype=np.uint8)
                Y[:, index] = annotation
                X[index] = im

    return X, Y


def generate_random_cut(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = rescut(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_brightness(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = change_brightness(
                X[index], Y[index], value=np.random.randint(15, 45), sign=True)
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_inversed_color(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = inverse_color(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_low_gamma(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = change_brightness(
                X[index], Y[index], value=np.random.randint(15, 45), sign=False)
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_night_effect(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = night_effect(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_horizontal_flip(Dataset, names2index, flipable_components, X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = list(Y)

    for index in range(len(X)):
        if indexes[index]:
            im, Y_aug = horizontal_flip(
                Dataset, names2index, flipable_components, X[index], Y_aug, index)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_random_shadows(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = add_random_shadow(X[index], Y[index])
            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_chained_transformations(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = change_brightness(X[index], Y[index])
            im, angle = add_random_shadow(im, angle)

            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug


def generate_random_noise(X, Y, proportion=0.25):
    indexes = np.random.choice([True, False], len(X), p=[
                               proportion, 1-proportion])

    X_aug = []
    Y_aug = []
    for index in range(len(X)):
        if indexes[index]:
            im, angle = add_rdm_noise(X[index], Y[index])

            Y_aug.append(angle)
            X_aug.append(im)

    return X_aug, Y_aug
