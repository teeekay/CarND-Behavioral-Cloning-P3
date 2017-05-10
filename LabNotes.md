May 9 2017
redid training -=
12:20-12:23 - 2 laps anti-clockwise
12:27-12:29 1 lap anti-clockwise
Beginning at start line
12:33 - move to center - bad (should remove)
12:34 - move to center from Left - good
12:35 - move to center from Right - good
12:36 - move to center from Left - OK
12:37 - Right Corner to Center on stripe corner#1 OK
12 38 - Right Corner to Center on stripe corner#1 OK
12 39 - Right Corner to Center on stripe corner#1 OK
12:40 - End of bridge to center of curve#2
12:41 - End of bridge to center of curve#2


tried to equalize histogram so not so weighted to center. Not great results

21660/21660 [==============================] - 29s - loss: 0.2552 - val_loss: 0.1797
Epoch 2/10
21660/21660 [==============================] - 26s - loss: 0.2258 - val_loss: 0.1665
Epoch 3/10
21660/21660 [==============================] - 26s - loss: 0.2174 - val_loss: 0.1594
Epoch 4/10
21660/21660 [==============================] - 26s - loss: 0.2102 - val_loss: 0.1546
Epoch 5/10
21660/21660 [==============================] - 26s - loss: 0.2078 - val_loss: 0.1511
Epoch 6/10
21660/21660 [==============================] - 26s - loss: 0.2024 - val_loss: 0.1483
Epoch 7/10
21660/21660 [==============================] - 26s - loss: 0.2001 - val_loss: 0.1460
Epoch 8/10
21660/21660 [==============================] - 26s - loss: 0.1979 - val_loss: 0.1442
Epoch 9/10
21660/21660 [==============================] - 26s - loss: 0.1953 - val_loss: 0.1425
Epoch 10/10
21660/21660 [==============================] - 26s - loss: 0.1950 - val_loss: 0.1411
dict_keys(['loss', 'val_loss'])
Model saved
9/9 [==============================] - 0s
test:00 :Path:./TestDrive2/IMG/left_2017_05_09_14_14_52_359.jpg Predicted value -0.256, Truth:0.300 Predicted Angle:-6.391 Truth: 7.500
test:01 :Path:./TestDrive2/IMG/center_2017_05_09_12_32_56_701.jpg Predicted value -0.089, Truth:-0.650 Predicted Angle:-2.219 Truth: -16.250
test:02 :Path:./TestDrive2/IMG/center_2017_05_09_14_16_24_796.jpg Predicted value 0.075, Truth:0.729 Predicted Angle:1.886 Truth: 18.233
test:03 :Path:./TestDrive2/IMG/right_2017_05_09_12_22_42_103.jpg Predicted value -0.146, Truth:-0.202 Predicted Angle:-3.643 Truth: -5.056
test:04 :Path:./TestDrive2/IMG/left_2017_05_09_17_21_23_983.jpg Predicted value -0.430, Truth:0.300 Predicted Angle:-10.755 Truth: 7.500
test:05 :Path:./TestDrive2/IMG/left_2017_05_09_17_19_35_434.jpg Predicted value 0.098, Truth:0.300 Predicted Angle:2.445 Truth: 7.500
test:06 :Path:./TestDrive2/IMG/right_2017_05_09_12_22_07_026.jpg Predicted value -0.219, Truth:-0.300 Predicted Angle:-5.487 Truth: -7.500
test:07 :Path:./TestDrive2/IMG/left_2017_05_09_17_26_10_540.jpg Predicted value -0.265, Truth:0.142 Predicted Angle:-6.629 Truth: 3.553
test:08 :Path:./TestDrive2/IMG/center_2017_05_09_12_22_29_284.jpg Predicted value -0.456, Truth:-0.008 Predicted Angle:-11.402 Truth: -0.188

tricked by water body - added runs veering away from water.

widened.

bolstered at problem areas - adjusted learning rate and decay to be 
``` python
model.compile(loss='mse', optimizer=Adam(lr=5e-5,decay=0.020))
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=20, callbacks=[checkpoint],validation_data=(X_val,y_val), shuffle=True)
```

``` sh
(carnd-term1) C:\Users\tknight\Source\Repos\CarND\CarND-Behavioral-Cloning-P3>python train_to_drive.py
Using TensorFlow backend.
kerasversion: 1.2.1
Starting Model
loaded 28768 images and measurements
loaded 7141 images and measurements
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 300, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 35, 150, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 35, 150, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 18, 75, 36)    21636       elu_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 18, 75, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 36, 48)     43248       elu_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 7, 36, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 17, 64)     27712       elu_3[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 17, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 15, 64)     36928       elu_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 15, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 15, 64)     0           elu_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 960)           0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           480500      flatten_1[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 500)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       elu_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           elu_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 20)            2020        dropout_2[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 20)            0           dense_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 20)            0           elu_8[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             21          dropout_3[0][0]
====================================================================================================
Total params: 663,989
Trainable params: 663,989
Non-trainable params: 0
____________________________________________________________________________________________________
samples_per_epoch comes out as 23913
nb_val_samples comes out as 5979
Train on 28768 samples, validate on 7141 samples
Epoch 1/20
2017-05-09 23:29:30.437703: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.439592: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.440825: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.442111: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.442984: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.443713: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.444203: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.445143: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-05-09 23:29:30.859297: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:887] Found device 0 with properties:
name: GeForce GTX 1060 6GB
major: 6 minor: 1 memoryClockRate (GHz) 1.7085
pciBusID 0000:01:00.0
Total memory: 6.00GiB
Free memory: 4.99GiB
2017-05-09 23:29:30.859582: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:908] DMA: 0
2017-05-09 23:29:30.861070: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:918] 0:   Y
2017-05-09 23:29:30.861764: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0)
28768/28768 [==============================] - 36s - loss: 0.1651 - val_loss: 0.1104
Epoch 2/20
28768/28768 [==============================] - 31s - loss: 0.1380 - val_loss: 0.1027
Epoch 3/20
28768/28768 [==============================] - 31s - loss: 0.1318 - val_loss: 0.0987
Epoch 4/20
28768/28768 [==============================] - 31s - loss: 0.1289 - val_loss: 0.0962
Epoch 5/20
28768/28768 [==============================] - 31s - loss: 0.1262 - val_loss: 0.0942
Epoch 6/20
28768/28768 [==============================] - 32s - loss: 0.1223 - val_loss: 0.0927
Epoch 7/20
28768/28768 [==============================] - 31s - loss: 0.1213 - val_loss: 0.0915
Epoch 8/20
28768/28768 [==============================] - 32s - loss: 0.1201 - val_loss: 0.0905
Epoch 9/20
28768/28768 [==============================] - 43s - loss: 0.1178 - val_loss: 0.0897
Epoch 10/20
28768/28768 [==============================] - 32s - loss: 0.1177 - val_loss: 0.0888
Epoch 11/20
28768/28768 [==============================] - 32s - loss: 0.1163 - val_loss: 0.0881
Epoch 12/20
28768/28768 [==============================] - 32s - loss: 0.1153 - val_loss: 0.0874
Epoch 13/20
28768/28768 [==============================] - 32s - loss: 0.1153 - val_loss: 0.0868
Epoch 14/20
28768/28768 [==============================] - 32s - loss: 0.1147 - val_loss: 0.0863
Epoch 15/20
28768/28768 [==============================] - 32s - loss: 0.1149 - val_loss: 0.0858
Epoch 16/20
28768/28768 [==============================] - 31s - loss: 0.1137 - val_loss: 0.0853
Epoch 17/20
28768/28768 [==============================] - 31s - loss: 0.1127 - val_loss: 0.0849
Epoch 18/20
28768/28768 [==============================] - 31s - loss: 0.1117 - val_loss: 0.0845
Epoch 19/20
28768/28768 [==============================] - 31s - loss: 0.1109 - val_loss: 0.0841
Epoch 20/20
28768/28768 [==============================] - 31s - loss: 0.1119 - val_loss: 0.0838
dict_keys(['loss', 'val_loss'])
Model saved
9/9 [==============================] - 0s
test:00 :Path:./TestDrive2/IMG/left_2017_05_09_14_16_58_508.jpg Predicted value -0.351, Truth:-0.444 Predicted Angle:-8.784 Truth: -11.109
test:01 :Path:./TestDrive2/IMG/center_2017_05_09_14_14_53_602.jpg Predicted value -0.061, Truth:-0.346 Predicted Angle:-1.519 Truth: -8.647
test:02 :Path:./TestDrive2/IMG/left_2017_05_09_17_18_49_470.jpg Predicted value 0.330, Truth:0.578 Predicted Angle:8.247 Truth: 14.455
test:03 :Path:./TestDrive2/IMG/center_2017_05_09_14_17_10_597.jpg Predicted value -0.406, Truth:-0.368 Predicted Angle:-10.145 Truth: -9.211
test:04 :Path:./TestDrive2/IMG/left_2017_05_09_17_20_09_928.jpg Predicted value 0.096, Truth:0.300 Predicted Angle:2.404 Truth: 7.500
test:05 :Path:./TestDrive2/IMG/center_2017_05_09_23_08_47_650.jpg Predicted value -0.020, Truth:-0.150 Predicted Angle:-0.503 Truth: -3.759
test:06 :Path:./TestDrive2/IMG/center_2017_05_09_22_21_57_191.jpg Predicted value -0.269, Truth:-0.594 Predicted Angle:-6.719 Truth: -14.850
test:07 :Path:./TestDrive2/IMG/center_2017_05_09_14_17_32_841.jpg Predicted value 0.098, Truth:-0.015 Predicted Angle:2.448 Truth: -0.376
test:08 :Path:./TestDrive2/IMG/left_2017_05_09_23_08_52_615.jpg Predicted value 0.121, Truth:0.217 Predicted Angle:3.018 Truth: 5.432
```
git commit at this point!
to work on next - a particular wobble - add images to fix?

then work on simplifying model

1D?
reduced dimensions of image
less layers

look ahead to next steer? to use in smoothing steer?
