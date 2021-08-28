from custom_modules import architectures

model_path = 'test_model\\models\\auto_label6.h5'
out = 'test_model\\models\\auto_label6.tflite'
architectures.keras_model_to_tflite(model_path, out)
