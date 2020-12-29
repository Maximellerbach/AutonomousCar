import matplotlib.pyplot as plt
import numpy as np


def plot_points(points, title="", show=True):
    points = np.array(points)
    assert len(points.shape) == 2
    assert points.shape[1] >= 2

    X = points[:, 1]
    Y = points[:, 0]
    p = plt.scatter(X, Y)
    p.axis('equal')
    p.title(title)
    if show:
        p.show()
    return p


def plot_bars(dict, width, title="", show=True):
    plt.bar(list(dict.keys()),
            list(dict.values()),
            width=width)
    plt.title(title)
    if show:
        plt.show()


def plot_time_series(X, Y, title="", show=True):
    x_len, y_len = len(X), len(Y)
    assert x_len == y_len

    its = [i for i in range(x_len)]
    p = plt.plot(its, X, Y)
    p.title(title)
    if show:
        p = plt.show()
    return p
