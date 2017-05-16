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


'''
This version of Drive.py has been modified to use small image sizes reduced by
a factor of 4 and converted to HSV.
'''
average_steer = True # try to smooth steering

use_RGB = False # use RGB images
use_V_only = False # use only V channel of HSV
reverse_throttle = False # Try braking to prevent oversteer
moderate_steer = False # try moderating steering to prevent oversteer


'''
set up global clahe so that it doesn't get repeated, and variables remain constant
'''
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))

'''
initialize lists to store recorded data in - this prevents recording from affecting
performance during autonomous driving, these can be saved when drive is complete.
'''
filenames = []
rec_images = []

# gloabal params for moderating steering
steer_history = [0.0,0.0,0.0,0.0,0.0,0.0]
steer_mod_weights = [1.4,2.0,0.6,0.3,0.15,0.08]
# increasing this should reduce steering angles - this could result in leaving the track on corners
straightening_factor = 1.0
'''
steer_mod
subroutine to try to moderate steering variability
'''
def steer_mod(new_steering_angle):
    steer_history[0] = new_steering_angle
    projected_next_steer = (steer_history[0] - steer_history[1]) + steer_history[0]
    steer_history.insert(0,projected_next_steer)
    steer_history.pop()
    adjusted_steering_angle = (sum(np.asarray(steer_history)*np.asarray(steer_mod_weights))/
        sum(np.asarray(steer_mod_weights)*straightening_factor))
    steer_history[0] = adjusted_steering_angle
    sys.stdout.write("Adjusting steer to {0:04.3f}\t".format(adjusted_steering_angle*90/3.14*0.872))
    return adjusted_steering_angle

'''
PI controller used for speed control
'''
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

'''
30.3 is maximum speed at which car will go
'''
set_speed = 30.3 #25#30.3 #10
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

        ## Added code to modify image sent to prediction
        if use_RGB == False:
            '''
            this is a bug which I introduced by mistake, and only just discovered
            I had put in - should be RGB2HSV_FULL since it is loaded by PIL Image.open
            but actually model works better this way at full speed. so
            I have left it in for now.  But it probably explains why car avoids
            shadows!
            '''
            image_array = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV_FULL)
            h,s,v = cv2.split(image_array)

            # I had hoped that image contrast enhancement might help problems in
            # shadows, but it does not appear to be a general solution
            cl_v = clahe.apply(v)
            if use_V_only == True:
                image_array = np.reshape(cl_v,(160,320,1))
            else:
                image_array = cv2.merge((h,s,cl_v))
                image_array = cv2.resize(image_array, (80,40), interpolation =  cv2.INTER_AREA)
        else:
            #image_array = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            image_array = cv2.resize(image2, (80,40), interpolation =  cv2.INTER_AREA)

        #determine steering angle based on model
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        #determine throttle to apply to get to set_speed with PI controller
        throttle = controller.update(float(speed))

        '''
        attempt to compensate for overlarge steering angles in training data by
        reducing the angle applied - not needed after switching to new training
        dataset. Moderate results - then used this formula to readjust training data at
        source.
        '''
        if abs(steering_angle) > 0.1 and moderate_steer == True:
            adjusted_angle = steering_angle/(abs(steering_angle)) * (0.1 + ((abs(steering_angle) -.1) * .75 * math.cos(steering_angle*math.pi/4.0)))
            sys.stdout.write("Adjusting steering to {0:02.3f} ".format(adjusted_angle))
            steering_angle = adjusted_angle

        '''
        method to prevent crashing at higher speeds by applying brakes when steering
        sharper turns (with original training dataset).  relatively succesful on track1
        '''
        if abs(steering_angle)>0.25 and float(speed)>24.0 and reverse_throttle== True:
            throttle = -15
            print("slowing down!")

        '''
        The next lines compensate for the fact that the PI controller doesn't always
        kick in at the start of a run.
        '''
        if float(speed)>(set_speed+0.5):
            throttle = 0.0

        sys.stdout.write("Calculated steering angle:{0:04.3f} \tthrottle: {1:04.2f} \t".format(steering_angle*90/3.14*0.872, throttle))

        '''
        This calls the steer_mod function to try to moderate the steering based
        on averaging a weighted history of previous steers and one projection forward
        '''
        if average_steer == True:
            steering_angle = steer_mod(steering_angle)

        sys.stdout.write("\r")
        sys.stdout.flush()

        '''
        send directions to the simulator
        '''
        send_control(steering_angle, throttle)

        '''
        if we're saving a history of the frames and angles, store it in a list
        which we will dump to file after the run so we have no effect on performance
        '''
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            filenames.append(image_filename)
            rec_images.append(pil_image)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

'''
client is disconnecting - now is the time to write out any frames stored during
the run.
'''
@sio.on('disconnect')
def disconnect(sid):
    print("In Disconnect")
    if args.image_folder != '':
        print("\n*******")
        if len(filenames) > 0:
            for i in range(len(filenames)):
                sys.stdout.write("Writing:{}\r".format(filenames[i]))
                sys.stdout.flush()
                rec_images[i].save('{}.jpg'.format(filenames[i]))
        print("\n*******")
    print("\nGoodbye")


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
