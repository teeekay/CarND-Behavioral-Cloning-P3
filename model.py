from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Dropout, Activation, Input
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras import __version__ as keras_version

import sklearn
from sklearn.model_selection import train_test_split

import cv2
import csv
import sys
import numpy as np
from random import shuffle, randint

''' define which datasets to use '''
use_Udacitydata = False # Data provided by Udacity
use_track1 = False #Data recorded on Track 1
use_track1_fixes = False # Data recorded in problem areas on Track 1
use_track2 = True # Data recorded on Track 2


print("Starting Training System")
print("kerasversion: {}".format(keras_version))

'''
Set model up to use small images (40,80,3) in HSV space
'''
use_RGB = False
use_small = True

''' clahe image enhancement defined globally '''
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))

''' Dataset paths '''
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

###############################################################################
'''
Define the Keras CNN
Based on the nVidia model for predicting steering angles
reduced number of planes, and size of input image
'''

model = Sequential()

''' normalize the input data '''
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(40,80,3)))
''' crop off distracting/extraneous top and edges '''
model.add(Cropping2D(cropping=((15,5), (3,3))))

'''Convolution layer 1'''
model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='same'))
model.add(ELU())

'''Convolution layer 2'''
model.add(Convolution2D(24, 3, 3, subsample=(1, 1), border_mode='same'))
model.add(ELU())

'''Convolution layer 3'''
model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='valid'))
model.add(ELU())

'''Convolution layer 4'''
model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='valid'))
model.add(ELU())

'''Convolution layer 5'''
model.add(Convolution2D(24, 3, 3, border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.35))

model.add(Flatten())

''' Fully connected Layer 1 '''
model.add(Dense(100))
model.add(ELU())

''' Fully connected Layer 2 '''
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.25))

''' Fully connected Layer 3 '''
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(0.25))

''' Fully connected Layer 4 '''
model.add(Dense(1))

''' Print out a summary of the Model '''
model.summary()

###############################################################################
'''
load in the training data
'''
samples = []
views = ['center_', 'left_', 'right_']
# angle offsets generated empirically for right and left views
# where 1.0 = 25 degrees, so 0.2 = 5 degrees
view_offsets = [0.0,0.2,-0.2]

'''
load up all the information from the specified csvfiles, including left and right
views of the drive
'''
for data_dir in train_dir:
    csv_path = data_dir + csv_filename
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            val =[]
            for i in range(len(views)):
                val =[line[0+i],float(line[3])+view_offsets[i]]
                samples.append(val)


# allocate 20% of the samples to be used for validation testing
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


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
        [data_dir, img_dir, filename] = sample[0].split('\\')[-3:]
        if use_small == False:
            local_path = './'+ data_dir + '/' + img_dir + '/' + filename
            image = cv2.imread(local_path)
            img = cv2.resize(image, (80,40), interpolation =  cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
            h,s,v = cv2.split(img)
            cl_v = clahe.apply(v)
            cl_img = cv2.merge((h,s,cl_v))
        else:
            img_dir = 'Small_IMG'
            local_path = './'+ data_dir + '/' + img_dir + '/' + filename
            cl_img = cv2.imread(local_path)
        images.append(cl_img)
        measurement = float(sample[1])
        '''make sure measurements stay in the range -1 to 1 (some
        synthetically created angles could be outside this range) '''
        if abs(measurement) > 1.0:
            measurement = 1.0*(measurement/abs(measurement))
        angles.append(measurement)
        ''' emphasize the images with sharper turns and attempt to compensate
        for curves being in one direction due to oval track '''
        if measurement < -0.4 or measurement > 0.3:
            fl_img = cv2.flip(cl_img, 1)
            measurement *= -1.0
            images.append(fl_img)
            angles.append(measurement)
        ''' progress indicator '''
        if totcntr % int(num_samples/1000) == 0:
            sys.stdout.write("Progress: {} % \t {} samples processed. \r"
                .format(int(1000*totcntr/num_samples)/10, totcntr))
            sys.stdout.flush()
    print()
    X_1 = np.array(images)
    y_1 = np.array(angles)
    print("loaded {} images and measurements".format(len(angles)))
    return(sklearn.utils.shuffle(X_1, y_1))

'''
Initially used generator, but switched back to preprocessing
attempted to move changes into generator, but have not tested.
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while(1):
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                [data_dir, img_dir, filename] = sample[0].split('\\')[-3:]
                if use_small == False:
                    local_path = './'+ data_dir + '/' + img_dir + '/' + filename
                    image = cv2.imread(local_path)
                    img = cv2.resize(image, (80,40), interpolation =  cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
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
                if measurement < -0.4 or measurement > 0.3:
                    fl_img = cv2.flip(cl_img, 1)
                    measurement *= -1.0
                    images.append(fl_img)
                    angles.append(measurement)
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


print("Starting to train Model")

X_train, y_train = prep_samples(train_samples)
X_val, y_val = prep_samples(validation_samples)

'''
set up model to use
Adam optimizer,
learning rate set at 0.00005 to start with decay rate of 2% per update
combined with dropout gives good loss curves on models
'''
model.compile(loss='mse', optimizer=Adam(lr=5e-5,decay=0.020))

print("Training samples per epoch = {}.".format(len(train_samples)))
print("Validation samples per epoch = {}.".format(len(validation_samples)))

'''
set up callbacks to save the model each epoch, and to stop running the model early
if validation loss falls below 0.0002 between epochs
'''
checkpoint = ModelCheckpoint('last_model{epoch:02d}.h5', monitor='lr',
    verbose=2)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=0,
    verbose=2, mode='auto')

'''
Run the model for up to 40 epochs unless stopped by earlystop callback
'''
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=40,
    callbacks=[checkpoint, earlystop], validation_data=(X_val,y_val),
    shuffle=True)

print()
model.save('./last_model.h5')
print("Model saved")


'''
plot the loss for training and validation sets vs epochs
'''
print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


'''
 pick first 20 samples from the validation set and visualize what the model
 is predicting
'''
X_test = X_val[0:20]
angles = y_val[0:20]
steers =[]

steers = model.predict(X_test,verbose=1)

for i in range(len(steers)):
    print("#{0:02d} Predicted Steer:{1:03.3f}, Actual:{2:03.3f} \
        Predicted Angle:{3:04.3f} Actual:{4:04.3f}"
        .format( int(i), float(steers[i]), float(angles[i]),
        float(steers[i])*25, float(angles[i]*25)))

    img = X_test[i]
    if use_RGB == False:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
    img = cv2.resize(img,None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
    '''
    plot steering angles on the images
    - red for prediction
    - green for recorded angle during training
    '''
    h,w = img.shape[0:2]
    cv2.line(img,(int(w/2),int(h)),
        (int(w/2+angles[i]*w/4),int(h/2)),(0,255,0),thickness=4)
    cv2.line(img,(int(w/2),int(h)),
        (int(w/2+steers[i]*w/4),int(h/2)),(0,0,255),thickness=4)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./last_model_"+str(i)+".png", img)

exit()
