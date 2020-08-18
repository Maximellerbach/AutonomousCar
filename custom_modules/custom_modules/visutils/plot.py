import matplotlib.pyplot as plt
import numpy as np

def plot_points(positions):

    positions = np.array(positions)
    X = positions[:, 1]
    Y = positions[:, 0]
    plt.scatter(X, Y)
    plt.axis('equal')
    plt.show()
