from flask import Flask
from takepicture import camera
import os

app = Flask(__name__)

@app.route('/image.jpg')

def image():
    cam.TakePicture()
    return app.send_static_file('image.jpg')

if __name__ == '__main__':
    # initialize the camera
    cam = camera()
    # run flask, host = 0.0.0.0 needed to get access to it outside of the host
    app.run(host='0.0.0.0',port=1337)