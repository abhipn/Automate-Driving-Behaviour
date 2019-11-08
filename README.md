## Automate Driving Behaviour

<img src="https://github.com/abhipn/Automate-Driving-Behaviour/blob/master/visualize.gif" height="680" width="420">

### Project Description
In this project, I've used convolutional neural networks to clone driving behavior. This model will output a steering angle to an autonomous vehicle. A lot of inspiraion for this model was taken from [Udacity Self driving car](https://github.com/udacity/CarND-Behavioral-Cloning-P3) module as well [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) model from NVIDIA.

### Dataset 
Approximately 63,000 images, 3.1GB. Data was recorded by [SullyChen](https://github.com/SullyChen/) around Rancho Palos Verdes and San Pedro California.

Download the dataset [here](https://github.com/SullyChen/driving-datasets) and extract the files into main directory.

### Files included
- `train_model.py` The script used for training the model.
- `helper.py` The script used for image processing and augmentation.
- `model.h5` The model weights. (i.e ModelCheckpoint)
- `visualize_test.py` The script for visualizing the prediction.

### Requirements
- You can install all required dependencies with pip install requirements.txt (or) conda install --file requirements.txt

### Model Architecture Design

I've used transfer learning approach, to build a Hybrid model. The bottom part of the model is based on of VGG16 which was pre-trained on ImageNet dataset. The output from first two non-trainable convolutional blocks of VGG16 is then connected to two trainable convolutional layers, three fully connected layers and a output layer. 

Here's the architecture of the model,

|Layer (type)                | Output Shape             | Param #    |
|----------------------------|--------------------------|------------|
|input_1 (InputLayer)        | [(None, 66, 200, 3)]     | 0          |
|block1_conv1 (Conv2D)       | (None, 66, 200, 64)      | 1792       |
|block1_conv2 (Conv2D)       | (None, 66, 200, 64)      | 36928      |
|block1_pool (MaxPooling2D)  | (None, 33, 100, 64)      | 0          |
|block2_conv1 (Conv2D)       | (None, 33, 100, 128)     | 73856      |
|block2_conv2 (Conv2D)       | (None, 33, 100, 128)     | 147584     |
|block2_pool (MaxPooling2D)  | (None, 16, 50, 128)      | 0          |
|conv2d (Conv2D)             | (None, 5, 22, 512)       | 3211776    |
|conv2d_1 (Conv2D)           | (None, 1, 18, 256)       | 3277056    |
|dropout (Dropout)           | (None, 1, 18, 256)       | 0          |
|flatten (Flatten)           | (None, 4608)             | 0          |
|dense (Dense)               | (None, 256)              | 1179904    |
|dense_1 (Dense)             | (None, 128)              | 32896      |
|dense_2 (Dense)             | (None, 64)               | 8256       |
|dense_3 (Dense)             | (None, 1)                | 65         |
|                            |**Total params**          |7,970,113   |


### Quick Start

1) First, install all the required dependencies from `requirements.txt` then download and extract the dataset into the main directory.
2) Now run `python train_model.py`for training the model. This will output a file `model-best.h5` (i.e epoch with least MSE on validation set).
3) Then test it on test set `python visualize.py` and visualize the results. 

### References:
 
 - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
 - Mariusz Bojarski, Philip Yeres, Anna Choromanska, Krzysztof Choromanski, Bernhard Firner, Lawrence Jackel, 
 Urs Muller. [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)
 - [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 
