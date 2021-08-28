from custom_modules import architectures

model_path = 'test_model\\models\\auto_label4.h5'
out = 'test_model\\models\\auto_label4.tflite'
architectures.keras_model_to_tflite(model_path)
