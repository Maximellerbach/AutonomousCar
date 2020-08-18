from keras.models import load_model

def quiver(model_path, input_folder):
    # see : https://github.com/keplr-io/quiver
    from quiver_engine import server
    model = load_model(model_path)
    server.launch(model, input_folder=input_folder, verbose=0)

