from custom_modules import architectures
import numpy as np

model_path = "test_model\\models\\auto_label7.h5"
out = "test_model\\models\\auto_label7.tflite"
architectures.keras_model_to_tflite(model_path, out)

# model = architectures.TFLite("C:\\Users\\maxim\\GITHUB\\AutonomousCar\\test_model\\models\\auto_label7.tflite")

# img = np.array(np.random.random((1, 1, 120, 160, 3)), dtype=np.float32)
# pred, dt = model.predict(img)

# print(pred, dt)
