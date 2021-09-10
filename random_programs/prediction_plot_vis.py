import os
from collections import deque

import matplotlib.pyplot as plt
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab
from matplotlib.animation import FuncAnimation

base_path = os.path.expanduser("~") + "\\random_data"
dos = f"{base_path}\\donkeycar\\"

physical_devices = tensorflow.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = []

model = architectures.safe_load_model("test_model\\models\\auto_label5.h5", compile=False)
architectures.apply_predict_decorator(model)

model2 = architectures.TFLite("test_model\\models\\auto_label5.tflite", output_names=["direction"])

gdos = Dataset.load_dataset_sorted(dos, flat=True)
model_outputs = architectures.get_model_output_names(model)

fig = plt.figure()
# plt.style.use("classic")
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
ax3 = ax1.twinx()

###
ax1.clear()
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlim(0, 200)
ax1.set_ylabel("ground_truth", color="blue")

ax2.clear()
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlim(0, 200)
ax2.set_ylabel("pred", color="red")

ax3.clear()
ax3.set_ylim(-1.0, 1.0)
ax3.set_xlim(0, 200)
ax3.set_ylabel("pred2", color="purple")

(line1,) = ax1.plot([], [], lw=1, color="blue")
(line2,) = ax2.plot([], [], lw=1, color="red")
(line3,) = ax3.plot([], [], lw=1, color="purple")

X = deque([0], maxlen=200)
Y = deque([0], maxlen=200)
Z = deque([0], maxlen=200)
Z2 = deque([0], maxlen=200)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    img, annotation = Dataset.load_img_and_annotation(gdos[i], to_list=False)

    to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)
    prediction_dict, elapsed_time = model.predict(to_pred)
    prediction_dict = prediction_dict[0]

    prediction_dict2, elapsed_time2 = model2.predict(to_pred)

    print(elapsed_time, elapsed_time2)

    vis_lab.vis_all_compare(Dataset, [1], img, annotation, prediction_dict)

    if len(X) < 200:  # fill the X list with numbers from 0 to 200
        X.append(X[-1] + 1)
    Y.append(annotation["direction"])
    Z.append(prediction_dict["direction"])
    Z2.append(prediction_dict2["direction"])

    line1.set_data(X, Y)
    line2.set_data(X, Z)
    line3.set_data(X, Z2)

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
