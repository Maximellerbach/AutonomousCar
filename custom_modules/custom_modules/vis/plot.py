import matplotlib.pyplot as plt
import numpy as np


def plot_points(points, ax1=None, title="", show=True):
    points = np.array(points)
    assert len(points.shape) == 2
    assert points.shape[1] >= 2

    if ax1 is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

    X = points[:, 1]
    Y = points[:, 0]
    ax1.scatter(X, Y)
    ax1.axis('equal')
    ax1.title.set_text(title)
    if show:
        plt.show()


def plot_bars(dict, width, ax1=None, title="", show=True):
    if ax1 is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

    ax1.bar(list(dict.keys()),
            list(dict.values()),
            width=width)
    ax1.title.set_text(title)
    if show:
        plt.show()


def plot_time_series(*args, ax1=None, title="", show=True):
    lenghts = [len(arg) for arg in args]
    assert lenghts == [lenghts[0]]*len(lenghts)
    its = [i for i in range(len(args[0]))]

    if ax1 is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

    ax1.clear()
    ax1.plot(its, args)
    ax1.title.set_text(title)
    if show:
        plt.show()
