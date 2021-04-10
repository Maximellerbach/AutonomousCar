import numpy as np
import matplotlib.pyplot as plt


def get_circuit_bound(prev_pos, now_pos):
    pos_vec = now_pos - prev_pos

    # get perpendicular vector to heading vector
    vWidth = np.cross(pos_vec, np.array([0, 1, 0]))
    vWidth = vWidth / np.linalg.norm(vWidth)

    left_point = now_pos + (vWidth * 3.25)
    right_point = now_pos - (vWidth * 3.25)

    return left_point, right_point


def plot_points(left_point, right_point):
    left_point = np.array(left_point)
    right_point = np.array(right_point)

    plt.scatter(left_point[:, 0], left_point[:, 2])
    plt.scatter(right_point[:, 0], right_point[:, 2])
    plt.show()


if __name__ == '__main__':

    point_path = "C:\\Users\\maxim\\GITHUB\\sim\\sdsandbox\\sdsim\\Assets\\Resources\\Track\\warehouse_path.txt"
    points = []

    with open(point_path, 'r') as f:
        for line in f:
            try:
                # considering: [x, y, z] : [width, height, depth]
                x, y, z = line.split(',')
                points.append(np.array([float(x), float(y), float(z)]))
            except:
                pass

    print(f"found {len(points)} points")

    left_bound = []
    right_bound = []
    for i in range(1, len(points)):
        l, r = get_circuit_bound(points[i-1], points[i])
        left_bound.append(l)
        right_bound.append(r)

    plot_points(left_bound, right_bound)
