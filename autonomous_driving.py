import base64
import socketio
import eventlet
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf

sio = socketio.Server(async_mode='eventlet')
app = Flask(__name__)
model = None

# def process_img(img):#model 2
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#   img = img[40:140]
#   img = cv2.resize(img, (200, 66))
#   img = img/127.5 - 1.0
#   #img = img**1.99999/255**1.99999 * 2 - 1
#   return img

def process_img(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = img[50:140]
    img = cv2.resize(img, (200, 66))
    img = img / 127.5 - 1.0
    return np.array([img])


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        viteza_curenta = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = process_img(image)

            unghi = float(model.predict(image))

            acc = 1.0 - unghi ** 2 - (viteza_curenta / 25) ** 2

            print("steering_angle:" + str(unghi))
            send_command(unghi, acc)
        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.event
def connect(sid, environ):
    print("connect ", sid)
    send_command(0, 0)


def send_command(unghi, acc):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(unghi),
            'throttle': str(acc)
        },
        skip_sid=True)


if __name__ == '__main__':
    model = load_model('model1.h5')
    print(model.summary())
    print(tf.__version__)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
