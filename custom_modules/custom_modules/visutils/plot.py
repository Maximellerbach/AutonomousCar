import matplotlib.pyplot as plt
import numpy as np


def plot_points(positions):
    positions = np.array(positions)
    assert len(positions.shape) == 2
    # at least 2D points (only 2 will be visualized anyway)
    assert positions.shape[1] >= 2

    X = positions[:, 1]
    Y = positions[:, 0]
    plt.scatter(X, Y)
    plt.axis('equal')
    plt.show()
