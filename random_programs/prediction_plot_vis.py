import os
from collections import deque

import matplotlib.pyplot as plt
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab
from matplotlib.animation import FuncAnimation

base_path = os.path.expanduser("~") + "\\random_data"
dos = f'{base_path}\\forza2\\'

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = [1]

model = architectures.safe_load_model(
    'test_model\\models\\forza4.h5', compile=False)
architectures.apply_predict_decorator(model)

gdos = Dataset.load_dataset_sorted(dos)
model_outputs = architectures.get_model_output_names(model)

fig = plt.figure()
plt.style.use("classic")
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

X = deque([], maxlen=200)
Y = deque([], maxlen=200)
Z = deque([], maxlen=200)


def animate(i):
    img, annotation = Dataset.load_img_and_annotation(gdos[0][i], to_list=False)

    to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)
    prediction_dict, elapsed_time = model.predict(to_pred, training=False)
    prediction_dict = prediction_dict[0]

    vis_lab.vis_all(Dataset, [1], img, prediction_dict)

    X.append(annotation['time'])
    Y.append(annotation['direction'])
    Z.append(prediction_dict['direction'])

    ax1.clear()
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("ground_truth", color="blue")
    ax1.plot(X, Y, color="blue")

    ax2.clear()
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_ylabel("prediction", color="red")
    ax2.plot(X, Z, color="red")


ani = FuncAnimation(fig, animate, interval=1)
plt.show()

# for dos in gdos:
#     prediction_dicts = {}
#     for output_name in model_outputs:
#         prediction_dicts[output_name] = []

#     for path in tqdm(dos):
#         img, annotation = Dataset.load_img_and_annotation(path)
#         to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)
#         prediction_dict, elapsed_time = model.predict(to_pred, training=False)
#         prediction_dict = prediction_dict[0]

#         for output_name in prediction_dict:
#             prediction_dicts[output_name].append(prediction_dict[output_name])

#         Y = prediction_dicts['direction']
#         plot.plot_time_series(Y, ax1=ax1, title='direction')

    # for output_name in prediction_dicts:
    #     X = [i for i in range(len(prediction_dicts[output_name]))]
    #     Y = prediction_dicts[output_name]
    #     plot.plot_time_series(X, Y, title=output_name)
