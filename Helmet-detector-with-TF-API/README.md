# Face mask Detector
Create a face mask detector using OpenCV, Keras/TensorFlow, and Deep Learning.
## Requirements
 - Tensorflow 2.0
 - Keras
 - numpy
 - cv2
 - imutils
 
## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow, cv2, numpy, imutils.

```bash
pip install tensorflow
```
```bash
pip install opencv-python
```
```bash
pip install numpy
```
```bash
pip install imutils
```

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

