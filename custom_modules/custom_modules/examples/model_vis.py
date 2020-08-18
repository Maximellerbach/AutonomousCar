from keras.models import load_model
from custom_modules.visutils.model_vis import quiver

if __name__ == "__main__":
    model = load_model("test_model\\models\\linear_trackmania.h5", compile=False)
    quiver(model, "C:\\Users\\maxim\\random_data\\1 ironcar driving\\")
