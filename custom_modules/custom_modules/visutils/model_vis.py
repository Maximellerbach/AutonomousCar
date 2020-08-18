from keras.models import load_model

def quiver(model, input_folder):
    # see : https://github.com/keplr-io/quiver
    from quiver_engine import server
    server.launch(model, input_folder=input_folder)

