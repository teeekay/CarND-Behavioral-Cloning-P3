import math
import sklearn
import cv2
import csv
import sys
import numpy as np

use_RGB = True

use_Udacitydata = False
use_track1 = False #True
use_track1_fixes = False
use_track2 = True

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

views = ['center_', 'left_', 'right_']
view_offsets = [0.0,0.2,-0.2]
samples =[]
for data_dir in train_dir:
    csv_path = data_dir + csv_filename
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            val =[]
            for i in range(len(views)):
                val =[line[0+i],float(line[3])+view_offsets[i]]
                samples.append(val)

totcntr = 0
num_samples = len (samples)
for sample in samples:
    totcntr += 1
    [data_dir, img_dir, filename] = sample[0].split('\\')[-3:] #get all to right of last /
    local_path = './'+ data_dir + '/' + img_dir + '/' + filename
    image = cv2.imread(local_path)
    img = cv2.resize(image, (80,40), interpolation =  cv2.INTER_AREA)
    if use_RGB == False:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) #same as NVIDIA - can I Lamda it - no!
        h,s,v = cv2.split(img)
        cl_v = clahe.apply(v)
        cl_img = cv2.merge((h,s,cl_v))
    else:
        cl_img = img
    local_small_path = './'+ data_dir + '/Small_IMG/' + filename
    cv2.imwrite(local_small_path,cl_img)
    if totcntr % int(num_samples/1000) == 0:
        sys.stdout.write("Progress: {} % \t {} samples processed. \r".format(int(1000*totcntr/num_samples)/10, totcntr))
        sys.stdout.flush()
