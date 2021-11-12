import os
from collections import deque

import matplotlib.pyplot as plt
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab
from matplotlib.animation import FuncAnimation

base_path = os.path.expanduser("~") + "\\random_data"
dos = f"{base_path}\\donkey\\"

Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = []

gdos = Dataset.load_dataset_sorted(dos, flat=True)

fig = plt.figure()
# plt.style.use("classic")
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
ax3 = ax1.twinx()

###
ax1.clear()
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlim(0, 200)
ax1.set_ylabel("direction", color="blue")

ax2.clear()
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlim(0, 200)
ax2.set_ylabel("speed", color="red")

ax3.clear()
ax3.set_ylim(-1.0, 1.0)
ax3.set_xlim(0, 200)
ax3.set_ylabel("throttle", color="purple")

(line1,) = ax1.plot([], [], lw=1, color="blue")
(line2,) = ax2.plot([], [], lw=1, color="red")
(line3,) = ax2.plot([], [], lw=1, color="purple")

X = deque([0], maxlen=200)
Y1 = deque([0], maxlen=200)
Y2 = deque([0], maxlen=200)
Y3 = deque([0], maxlen=200)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    img, annotation = Dataset.load_img_and_annotation(gdos[i], to_list=False)
    vis_lab.vis_all(Dataset, [], img, annotation)

    if len(X) < 200:  # fill the X list with numbers from 0 to 200
        X.append(X[-1] + 1)

    Y1.append(annotation["direction"])
    Y2.append(annotation["speed"])
    Y3.append(annotation["throttle"])

    line1.set_data(X, Y1)
    line2.set_data(X, Y2)
    line3.set_data(X, Y3)

    return line1, line2


ani = FuncAnimation(
    fig,
    animate,
    init_func=init,
    interval=1,
    frames=len(gdos),
)
# plt.draw()
plt.show()
