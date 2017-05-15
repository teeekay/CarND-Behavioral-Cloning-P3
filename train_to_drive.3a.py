from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Dropout, Activation, Input
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras import __version__ as keras_version

from sklearn.model_selection import train_test_split
import sklearn
import cv2
import csv
import sys
import numpy as np
from random import shuffle, randint
print("Starting Training System")
print("kerasversion: {}".format(keras_version))

use_Udacitydata = False
use_track1 = False #True
use_track1_fixes = False
use_track2 = True

use_RGB = True
use_small = True

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))

train_dir = []
if use_track1 == True:
    train_dir.append('./TestDrive2/')
if use_Udacitydata == True:
    train_dir.append('./data/')
if use_track2 == True:
    train_dir.append('./Track2/')
if use_track1_fixes == True:
    train_dir.append('./track1_fixes/')

csv_filename = 'driving_log.csv'
#img_root = root_dir + 'IMG/'
##############################################
# set up cropping2D layer
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(40,80,3)))
#model.add(Cropping2D(cropping=((70,20), (10,10))))
model.add(Cropping2D(cropping=((15,5), (3,3))))
#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same'))
model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='same'))
model.add(ELU())
#model.add(Dropout(0.1))
model.add(Convolution2D(24, 3, 3, subsample=(1, 1), border_mode='same'))
model.add(ELU())
#model.add(Dropout(0.2))
model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='valid'))
model.add(ELU())
#model.add(Dropout(0.25))
model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='valid'))
#model.add(Dropout(0.25))
model.add(ELU())
model.add(Convolution2D(24, 3, 3, border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(100)) #500
model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(50)) #100
model.add(ELU())
model.add(Dropout(0.25))
model.add(Dense(10)) #20
model.add(ELU())
model.add(Dropout(0.25))
model.add(Dense(1))

model.summary()

###############################################################################

samples = []
#load in datafile with steering measurements
#edited csv file to remove so many 0 steers
#with open('./TestDrives/driving_log1.csv') as csvfile:
views = ['center_', 'left_', 'right_']
view_offsets = [0.0,0.2,-0.2]

for data_dir in train_dir:
    csv_path = data_dir + csv_filename
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            val =[]
            for i in range(len(views)):
                val =[line[0+i],float(line[3])+view_offsets[i]]
                samples.append(val)


#but there is actually 3 times this much data since center, left and right cameras
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def displayCV2(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Bulk method
def prep_samples(samples):
    print ("starting preparation of images")
    num_samples = len(samples)
    shuffle(samples)
    images = []
    angles = []
    totcntr = 0
    for sample in samples:
        totcntr += 1
        [data_dir, img_dir, filename] = sample[0].split('\\')[-3:] #get all to right of last /
        if use_small == False:
            local_path = './'+ data_dir + '/' + img_dir + '/' + filename
            image = cv2.imread(local_path)
            img = cv2.resize(image, (80,40), interpolation =  cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) #same as NVIDIA - can I Lamda it - no!
            h,s,v = cv2.split(img)
            cl_v = clahe.apply(v)
            cl_img = cv2.merge((h,s,cl_v))
        else:
            img_dir = 'Small_IMG'
            local_path = './'+ data_dir + '/' + img_dir + '/' + filename
            cl_img = cv2.imread(local_path)
        images.append(cl_img)
        measurement = float(sample[1])
        #make sure measurements stay in the range -1 to 1 (some adjusted angles could be outside this range)
        if abs(measurement) > 1.0:
            measurement = 1.0*(measurement/abs(measurement))
        angles.append(measurement)
        # emphasize the images with sharper turns and attempt to compensate for curves being in one direction
        if measurement < -0.4 or measurement > 0.3:
            fl_img = cv2.flip(cl_img, 1)
            measurement *= -1.0
            images.append(fl_img)
            angles.append(measurement)
        if totcntr % int(num_samples/1000) == 0:
            sys.stdout.write("Progress: {} % \t {} samples processed. \r".format(int(1000*totcntr/num_samples)/10, totcntr))
            sys.stdout.flush()
    print()
    X_1 = np.array(images)
    y_1 = np.array(angles)
    print("loaded {} images and measurements".format(len(angles)))
    return(sklearn.utils.shuffle(X_1, y_1))

#not using - don't want to reprocess every time
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while(1):
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                [data_dir, img_dir, filename] = sample[0].split('\\')[-3:] #get all to right of last /
                if use_small == False:
                    local_path = './'+ data_dir + '/' + img_dir + '/' + filename
                    image = cv2.imread(local_path)
                    img = cv2.resize(image, (80,40), interpolation =  cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) #same as NVIDIA - can I Lamda it - no!
                    h,s,v = cv2.split(img)
                    cl_v = clahe.apply(v)
                    cl_img = cv2.merge((h,s,cl_v))
                else:
                    img_dir = 'Small_IMG'
                    local_path = './'+ data_dir + '/' + img_dir + '/' + filename
                    cl_img = cv2.imread(local_path)
                images.append(cl_img)
                measurement = float(sample[1])
                if abs(measurement) > 1.0:
                    measurement = 1.0*(measurement/abs(measurement))
                angles.append(measurement)
                # emphasize the images with sharper turns and attempt to compensate for curves being in one direction
                if measurement < -0.4 or measurement > 0.3:
                    fl_img = cv2.flip(cl_img, 1)
                    measurement *= -1.0
                    images.append(fl_img)
                    angles.append(measurement)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


print("Starting Model")

X_train, y_train = prep_samples(train_samples)
X_val, y_val = prep_samples(validation_samples)

model.compile(loss='mse', optimizer=Adam(lr=5e-5,decay=0.020))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
print("samples_per_epoch comes out as {}".format(len(train_samples)))
print("nb_val_samples comes out as {}".format(len(validation_samples)))

checkpoint = ModelCheckpoint('model3a_track2RGB{epoch:02d}.h5', monitor='lr',verbose=2)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=0, verbose=2, mode='auto')

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=40, callbacks=[checkpoint, earlystop], validation_data=(X_val,y_val), shuffle=True)


print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('./model3a_track2RGB.h5')
#model.save
print("Model saved")
#model.get_weights()
#print(validation_samples.shape)


X_test = X_val[0:20]
angles = y_val[0:20]

steers =[]
steers = model.predict(X_test,verbose=1)

for i in range(len(steers)):
    print("test:{:02d} Predicted value {:.3f}, Truth:{:.3f} Predicted Angle:{:.3f} Truth: {:.3f}"
        .format( int(i), float(steers[i]), float(angles[i]), float(steers[i])*25, float(angles[i]*25)))

    img = X_test[i]
    if use_RGB == False:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
    img = cv2.resize(img,None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angles[i]*w/4),int(h/2)),(0,255,0),thickness=4)
    cv2.line(img,(int(w/2),int(h)),(int(w/2+steers[i]*w/4),int(h/2)),(0,0,255),thickness=4)
    displayCV2(img)
    cv2.imwrite("./model3a_track2RGB"+str(i)+".png", img)

exit()
