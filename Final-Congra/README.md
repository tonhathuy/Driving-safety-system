# Helmet and mask detector
Create a face mask detector using OpenCV, Keras/TensorFlow, and Deep Learning.
## Requirements
 - Tensorflow 2.0
 - Keras
 - numpy
 - cv2
 - imutils
 - ~~playsound~~ -> pygame
## Installation
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
## 

## Face mask detection with OpenCV in images 
Open up a terminal, and execute the following command:
```bash
python detect_mask_image.py --image examples/example_01.png
```
![Image](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_result01.jpg)

As you can see, our face mask detector correctly labeled this image as Mask.

Let’s try another image, this one of a person not wearing a face mask:
```bash
python detect_mask_image.py --image examples/example_02.png 
```
![Image](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_result02.jpg)

Our face mask detector has correctly predicted No Mask.
## Detecting face masks with OpenCV in real-time
You can then launch the mask detector in real-time video streams using the following command:
```bash
python detect_mask_video.py
```
![Image](https://media.giphy.com/media/LLwnm6pcNu0HMwbqtA/giphy.gif)

Here, you can see that our face mask detector is capable of running in real-time (and is correct in its predictions as well).

## License
[MIT](https://choosealicense.com/licenses/mit/)

