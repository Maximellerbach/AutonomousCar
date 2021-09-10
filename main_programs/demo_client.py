import time

from custom_modules import architectures
from custom_modules.datasets import dataset_json

import sim_client


model_path = "test_model\\models\\test_scene.h5"
model = architectures.safe_load_model(model_path, compile=False)
architectures.apply_predict_decorator(model)
dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = [1]

client = sim_client.SimpleClient(("127.0.0.1", 9091), model, dataset, input_components, name="0")

time.sleep(1)
client.start()

while True:
    _, image = client.get_latest()
    prediction = client.predict(image)
    client.send_controls(prediction, 0.4, 0)
