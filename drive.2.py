import argparse
import base64
from datetime import datetime
import os
import sys
import shutil
import cv2
import math

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#use_S_only = True
use_Y_only = False
reverse_throttle = False # Try braking to prevent oversteer
moderate_steer = False # try moderating steering to prevent oversteer

clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32,32))

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 30
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        pil_image = Image.open(BytesIO(base64.b64decode(imgString)))
        image2 = np.asarray(pil_image)
        ## next line by tony
        image_array = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV_FULL)
        h,s,v = cv2.split(image_array)
        cl_v = clahe.apply(v)
        #image_array = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)
        #y,u,v = cv2.split(image_array)
        #cl_y = clahe.apply(y)
        if use_Y_only == True:
            #image_array = np.reshape(cl_y,(160,320,1))
            image_array = np.reshape(cl_v,(160,320,1))
            #print(image_array.shape)
        else:
            image_array = cv2.merge((h,s,cl_v))
        # to write whats going to model to video
        cv_image = image_array[70:140,10:310,:]

        #if use_Y_only == True:
        #    cv_image2 = np.reshape(cv_image,(70,310))
        #    cv_image = cv2.cvtColor(cv_image2, cv2.COLOR_GRAY2BGR)

        #pil_image = Image.fromarray(cv_image)
        #print("About to predict")
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        #steering_angle = steer_smoother(steering_angle)
        throttle = controller.update(float(speed))

        if abs(steering_angle) > 0.1 and moderate_steer == True:

            adjusted_angle = steering_angle/(abs(steering_angle)) * (0.1 + ((abs(steering_angle) -.1) * .75 * math.cos(steering_angle*math.pi/4.0)))
            #adjusted_angle = steering_angle/(abs(steering_angle)) * (0.1 + ((steering_angle -.1) * .6 * math.cos(steering_angle*math.pi/2.0)))
            print("adjusting steering from {} to {}".format(steering_angle, adjusted_angle))
            steering_angle = adjusted_angle

        if abs(steering_angle)>0.25 and float(speed)>24.0 and reverse_throttle== True:
            throttle = -15
            print("slowing down!")


        sys.stdout.write("steering angle:{0:02.3f}   \t\tthrottle: {0:.2f} \r".format(steering_angle*90/3.14*0.872, throttle))
        sys.stdout.flush()


        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            pil_image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
