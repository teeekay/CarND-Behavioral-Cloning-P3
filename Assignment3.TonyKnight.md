# **Behavioral Cloning Project** 

### Writeup by Tony Knight - 2017/05/15 

<img src="https://github.com/teeekay/CarND-Behavioral-Cloning-P3/blob/master/examples/Simulator.png?raw=true" alt="Udacity Simulator ready for training" width=600>

<u><i>Figure 1. Udacity Simulator ready for training</i></u>

---

#### 1. Submission files

My project includes the following files:
* adjust.py script to process training images into smaller size
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Assignment3.TonyKnight.md - this report
* final_run.mp4 - a video of the car driving around Track 1 autonomously in both directions.

These are all located at my [github repo](https://github.com/teeekay/CarND-Behavioral-Cloning-P3/)

#### 2. Submission includes functional code
The car can be driven autonomously around the track by executing :

```sh
python drive.py model.h5
```
while the Udacity provided simulator is in autonomous mode on Track 1.


#### 3. Submission

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 40, 80, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 20, 74, 3)     0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 20, 74, 24)    1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 20, 74, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 74, 24)    5208        elu_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 20, 74, 24)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 36, 24)     5208        elu_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 9, 36, 24)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 17, 24)     5208        elu_3[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 4, 17, 24)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 15, 24)     5208        elu_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 2, 15, 24)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2, 15, 24)     0           elu_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 720)           0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           72100       flatten_1[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           elu_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           elu_8[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]
====================================================================================================
Total params: 100,327
Trainable params: 100,327
```

<U><B>Table 1:</B><I> Keras Sequential Model Based on NVIDIA Architecture</I></U>

My model consists of a slightly modified version of NVIDIA's model convolution neural network  described in the document [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convolutional layers and 4 fully connected layers.  The model layers are set up on lines 58 to 103 of model.py.  

<i>Inline Image Processing (model.py lines 60-63).</i>  A Lamda layer was used to normalize the data to have a mean near 0.0 and range of (-1,1).  The image was also cropped to remove sky, car hood and extraneous features in the top, base, and sides of the image.  

<i>Convolutional Layers (model.py lines 65-84).</i>  The initial convolutional layer dimensions were reduced to 20 x 72 to accommodate the smaller cropped input images, and the number of planes was reduced to 24.  A 5x5 filter was used in the first convolutional layer with a stride of 1x1.  3X3 filters were used in the remaining convolutional layers with 1x1 strides in layers 2 and 5, and 2x2 strides in layers 3 and 4.  

<i>Fully Connected Layers (model.py lines 88-103).</i>  Four fully connected (Dense) Layers are activated with ELU

The model contains dropout layers in both the final convolutional layer and the 2nd and 3rd fully connected layers to help prevent overfitting (model.py lines 84, 95, 100). 

#### 3. Model parameter tuning

An Adam optimizer was used with the model.  The initial learning rate and decay rate were adjusted to 0.00005 and 2% respectively to produce best results based on previous test runs. (model.py line 237).

Callbacks were used at the end of each epoch to checkpoint the model, and to see if the training should be exited early (if evidence of overfitting).  Based on observing good training performance once the model and training data were set up, the early exit was configured to occur if validation loss decreased by less than 0.002 between epochs.

<img src="https://github.com/teeekay/CarND-Behavioral-Cloning-P3/blob/master/examples/Kerasrun.png?raw=true" alt="Udacity Simulator ready for training" width=600>

<u><i>Figure 2. Graph of Mean Squared Loss for Training and Validation Sets Suggests good Fit</i></u>


#### 4. Appropriate training data

<img src="https://github.com/teeekay/CarND-Behavioral-Cloning-P3/blob/master/examples/Track1center.png?raw=true" alt="Right, Center, and Left views of drive down down center line on Track 1" width=600>


I initially attempted to use training data produced by driving around track 1 twice counterclockwise, and once clockwise, trying to stay at the center of the road. 

I produced a set of data for areas of Track 1 where the car was steered away from the edge at sharp angles (and slow speeds) 

I also produced a set of data driving around track 2 (jungle track) several times in both directions at the center of the lane.

<img src="https://github.com/teeekay/CarND-Behavioral-Cloning-P3/blob/master/examples/Trainingexample.png?raw=true" alt="Driving down center line on Track 2" width=300>


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I attempted to copy the NVIDIA CNN architecture with the full 160x320 RGB image converted to YUV, and applying a clahe contrast equalization to the Y layer (to try to compensate for darker images) on the track.

I produced my own training data from track 1 driving in the center of the lane, and augmented the data by using the left and right camera images but the model generated mainly straight predictions and the car ran off the track at the first corner.  I attempted to equalize the dataset by removing a random selection of the large proportion of images with zero steering angle which dominated the set.  (I did this by randomly sorting the records with angles below 0.05 in excel, and removing a chunk of 2,000 records).

This worked better, but the model predictions failed in the simulator at specific locations which I noted.  I then produced datasets where I drove in those areas repeatedly and steered into the center of the road.  After training the car in this way, I was able to get the car to go autonomously around the track at 10 mph.  The car went off the track immediately on track 2.

I added a dataset of driving around track 2.  This helped further in equalizing the entire dataset, as there are many more sharp turns in both directions on track 2 requiring sharper steering angles, and less straight ahead driving.

As I tried to get the car to drive around the track faster, I found that it became unstable on straightaways, weaving at sharper and sharper angles.  I attempted to compensate for this by reducing the steer angle by a larger amount as the angle became larger (using the cosine of the angle to generate the reduction).  This was somewhat successful.  I hypothesized that the training set samples used to train the car away from the danger areas had been recorded with angles that were too sharp.  I used the same formula to adjust the angles in the training set and retrained the model.

The result was not significantly better.

I restarted at this point with an entirely new set of training data.  I drove around track 1 several times in each direction and drove smoother recovery sections in problem areas, I also added loops in each direction on track 2. I used the same model, and this was able to drive autonomously around the track at speeds of up to 27 mph.  I implemented a braking algorithm which would be activated in sharper turns at higher speeds (drive.py lines 163-164) which enabled the car to be set to go round the track at top speed, with some slowdowns in problem areas.

The new model was no better on Track 2.

I began investigating other aspects of the model.  I had seen a blog post by [Mengxi Wu](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234) about using a much smaller model with good results.  I tried switching to just using the Y layer (grayscale), but this did not produce equivalent results.  I reduced the number of planes in the layers to 24 in all the convolutional layers.  This did not make simulator performance any worse.  I then tried reducing the size of the images to 1/4 of the original size.  This significantly reduced training time.  the model performed about the same.

I wanted to see if I could get the model to work better on Track 2.  I got a fresh dataset from track 2 and drove several times around the track in both directions (I was getting better at controlling the simulator).  I trained the model using only the new track 2 dataset (augmented with right and left camera angles and flips of any images with sharper angles).  The model trained on a total of 63016 samples, and was validated on 15677 samples.  These included 17091 flipped images in the training set.  The model achieved a mse of 0.1024 after 17 epochs when early stopping was triggered.

The car crashed right away when tested on Track 2.  However, it performed better than my previous model on Track 1, being able to drive round the entire track at full speed.  This was interesting as it had never seen images from track 1 during training.  

I worked on implementing a queue-like data structure to hold a recent history of steering angles.  This was used to generate a rough prediction of the next steering angle, and then to produce a weighted average of the steering angle for the current timestep.  This slightly smoothed wobbling of the steering angle between steps, but could not prevent some weaving.  

The car was able to drive autonomously around Track 1 at full throttle (30 mph speed).  However, when I tried to record this feat, the car crashed in areas where it had previously navigated successfully.  I hypothesized that the saving of the image files was increasing the delay between steering predictions.  I modified the drive.py code to save the images into a list, and to save the images after the drive had been completed.  With this modification the car drove successfully around the track in both directions at full seed autonomously.

I realized about this point that the routine I was using to convert the images in drive.py was assuming a BGR input, whereas from the PIL Image fuction it would be in RGB.  I switched the routine to the correct way, but this resulted in crashes at top speed on Track 1 in places where it performed ok with the wrong routine.  The model no longer tried to avoid shadows on Track 1 which had been the case.  I put the code back to the way it had been previously as it was working that way and the deadline was near.

On track 2 the model moves toward curved trees or fence posts that are straight ahead in a curve, but may look like road features.  It is also fooled by other tracks it can see straight ahead even though the track is veering to the right or left.  To attempt make the model work on track 2, my next course of action would be to investigate if 
a) other image processing methods -possibly hough/sobel filters could be used in conjunction with the model to better delineate road boundaries and discriminate between road boundaries and other objects like trees and fence posts; b) investigate image transforms to eliminate the effect of shade in Track 2.

### Visualization of MOdel performance