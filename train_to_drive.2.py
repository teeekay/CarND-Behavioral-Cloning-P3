from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Dropout, Activation, Input
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback
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

use_Udacitydata = False #True
use_track2 = True
use_Y_only = False #True # try using only the greyscale
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32,32))


train_dir = []
train_dir.append('./TestDrive2/')
if use_Udacitydata == True:
    train_dir.append('./data/')
if use_track2 == True:
    train_dir.append('./Track2/')

csv_filename = 'driving_log.csv'
#img_root = root_dir + 'IMG/'


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

#not using - don't want to reprocess every time
def generator(samples, batch_size=32):
    #view_offsets = [0.0]
    num_samples = len(samples)

    while(1):
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split('\\')[-1] #get all to right of last /
                current_path = img_root + filename
                image = cv2.imread(current_path)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) #same as NVIDIA - can I Lamda it - no!
                y,u,v = cv2.split(img)
                cl_y = clahe.apply(y)
                cl_img = cv2.merge((cl_y,u,v))
                images.append(cl_img)
                measurement = float(batch_sample[1])
                if measurement > 1.0:
                    measurement = 1.0
                if measurement < -1.0:
                    measurement = -1.0
                angles.append(measurement)

                if abs(measurement) > 0.5:
                    image = cv2.flip(img, 1)
                    measurement *= -1.0
                    images.append(image)
                    angles.append(measurement)
                X_train = np.array(images)
                y_train = np.array(angles)

                #print("loaded {} images and measurements".format(len(angles)))
                yield sklearn.utils.shuffle(X_train, y_train)

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
        local_path = './'+ data_dir + '/' + img_dir + '/' + filename
        image = cv2.imread(local_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL) #same as NVIDIA - can I Lamda it - no!
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        #y,u,v = cv2.split(img)
        h,s,v = cv2.split(img)

        #cl_y = clahe.apply(y)
        cl_v = clahe.apply(v)
        if use_Y_only == True:
            #cl_img = np.reshape(cl_y, (160,320,1))
            cl_img = np.reshape(cl_s, (160,320,1))
            #cl_img = cv2.resize(cl_s, (80,160), interpolation =  cv2.INTER_AREA)
        else:
            #cl_img = cv2.merge((cl_y,u,v))
            cl_img = cv2.merge((h,s,cl_v))
        #print(cl_img.shape)
        if totcntr % int(num_samples/10) == 0:
            cv2.imwrite('./hsv_'+str(totcntr)+'_orig.png',image)
            cv2.imwrite('./hsv_'+str(totcntr)+'_v.png',v)
            cv2.imwrite('./hsv_'+str(totcntr)+'_vcl.png',cl_v)
            cv2.imwrite('./hsv_'+str(totcntr)+'_h.png',h)
            cv2.imwrite('./hsv_'+str(totcntr)+'_s.png',s)
            nw_img = cv2.cvtColor(cl_img, cv2.COLOR_HSV2BGR_FULL)
            cv2.imwrite('./hsv_'+str(totcntr)+'_mod.png',nw_img)
        images.append(cl_img)
        measurement = float(sample[1])
        if measurement > 1.0:
            measurement = 1.0
        if measurement < -1.0:
            measurement = -1.0
        angles.append(measurement)
        if abs(measurement) > 0.4:
            if use_Y_only == True:
                fl_img = cv2.flip(cl_y, 1)
                fl_img = np.reshape(fl_img, (160,320,1))
            else:
                fl_img = cv2.flip(cl_img, 1)
            measurement *= -1.0
            images.append(fl_img)
            angles.append(measurement)
        if totcntr % int(num_samples/1000) == 0:
            sys.stdout.write("Progress: {} % \t {} samples processed. \r".format(int(1000*totcntr/num_samples)/10, totcntr))
            sys.stdout.flush()

    X_1 = np.array(images)
    y_1 = np.array(angles)
    print("\nloaded {} images and measurements".format(len(angles)))
    return(sklearn.utils.shuffle(X_1, y_1))

print("Starting Model")

#train_generator = generator(train_samples, batch_size=128)
#validation_generator = generator(validation_samples, batch_size=128)
X_train, y_train = prep_samples(train_samples)
X_val, y_val = prep_samples(validation_samples)

# set up cropping2D layer
model = Sequential()
if use_Y_only == True:
    model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160,320,1)))
else:
    model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,20), (10,10))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same'))
model.add(ELU())
#model.add(Dropout(0.1))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same'))
model.add(ELU())
#model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(ELU())
#model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid'))
#model.add(Dropout(0.25))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(500))
model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.25))
model.add(Dense(20))
model.add(ELU())
model.add(Dropout(0.25))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer=Adam(lr=5e-5,decay=0.020))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
print("samples_per_epoch comes out as {}".format(len(train_samples)))
print("nb_val_samples comes out as {}".format(len(validation_samples)))

checkpoint = ModelCheckpoint('model_hsv{epoch:02d}.h5')

#history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#    validation_data=validation_generator, nb_val_samples=len(validation_samples),
#    nb_epoch=3, callbacks=[checkpoint])
# history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#    validation_data=validation_generator, nb_val_samples=len(validation_samples),
#    nb_epoch=10, callbacks=[checkpoint], verbose=2)
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=40, callbacks=[checkpoint], validation_data=(X_val,y_val), shuffle=True)


print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('./model_hsv.h5')
#model.save
print("Model saved")
#model.get_weights()
#print(validation_samples.shape)

sklearn.utils.shuffle(X_val, y_val)
tests = validation_samples[0:20]
#tests = np.random.choice(validation_samples, 20)
images =[]
angles = []
paths = []

for line in tests:
    filename = line[0].split('\\')[-1] #get all to right of last /
    current_path = img_root + filename
    paths.append(current_path)
    image = cv2.imread(current_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) #same as NVIDIA - can I Lambda it?  No!
    y,u,v = cv2.split(img)
    cl_y = clahe.apply(y)
    if use_Y_only == True:
        cl_img = np.reshape(cl_y, (160,320,1))
    else:
        cl_img = cv2.merge((cl_y,u,v))

    images.append(cl_img)
    measurement = float(line[1])
    angles.append(measurement)

X_test = np.array(images)

steers =[]
steers = model.predict(X_test,verbose=1)

for i in range(len(steers)):
    print("test:{:02d} :Path:{:s} Predicted value {:.3f}, Truth:{:.3f} Predicted Angle:{:.3f} Truth: {:.3f}"
        .format( int(i), str(paths[i]), float(steers[i]), float(angles[i]), float(steers[i])*25, float(angles[i]*25)))

    img = X_test[i]
    img = img[70:140,10:310,:] # crop to same size as during lambda layer
    if use_Y_only == True:
        img = np.reshape(img,(70,300))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        #img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
    img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angles[i]*w/4),int(h/2)),(0,255,0),thickness=4)
    cv2.line(img,(int(w/2),int(h)),(int(w/2+steers[i]*w/4),int(h/2)),(0,0,255),thickness=4)
    displayCV2(img)
    cv2.imwrite("./reducesteer_"+str(i)+".png", img)

exit()
