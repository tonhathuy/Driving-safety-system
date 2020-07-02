# Helmet and mask detector
Create a face mask detector using OpenCV, Keras/TensorFlow, and Deep Learning.
## Requirements
 - Tensorflow 2.0
 - Keras
 - numpy
 - cv2
 - imutils
 - ~~playsound~~ -> pygame
## [Installation](https://github.com/tonhathuy/Driving-safety-system/blob/master/Final-Congra/environment.yml)
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow, cv2, numpy, imutils.

install tensorflow
```bash
pip install tensorflow-gpu == 2.0
```
```bash
pip install opencv-python numpy imutils pygame
```
# Helmet detector
## I. Create dataset
### 1. Taking normal images of faces
[SCUT-FBP5500v2](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution), which allows different computational model with different facial beauty prediction paradigms, such as appearance-based/shape-based facial beauty classification/regression/ranking model for male/female of Asian/Caucasian.

### 2. Create Dataset Face Helmet
- first, search and download image Face helmet from google search image with tool [Download All Images](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en-US)
- cut head and helmet with [my tool](https://github.com/tonhathuy/Driving-safety-system/blob/master/Tool/cut_face_for_dataset.py)
```bash
python cut_face_for_dataset.py -d <directoryImageFolder> -o <directoryImageFolder after cut>
```
 - Now, we have a problem this is we have too little data for training models, so we used a technique to solve this problem. [Image data augmentation](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset. We used Data augmentation in Keras. The Keras deep learning library provides the ability to use data augmentation automatically when training a model.

## II training with Keras and TensorFlow
There are a number of important updates in TensorFlow 2.0, including eager execution, automatic differentiation, and better multi-GPU/distributed training support, but the most important update is that Keras is now the official high-level deep learning API for TensorFlow. We used Keras and TensorFlow to train a classifier to automatically detect whether a person is wearing a helmet or not.

## [DEMO](https://www.youtube.com/watch?v=WPadfu7S5rw)

