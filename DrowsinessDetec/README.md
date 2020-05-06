# DROWNSINESS DETECTION

This program is used to detect drowsiness for any given person. In this program we check how long a person's eyes have been closed for. If the eyes have been closed for a long period, the program will alert the user by playing an alarm sound.

### Prerequisites

1) install python
2) install pip
3) install opencv
4) install cmake
 -  install dlib: 
   ```
   pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
   ```
5) install playsound
6) install imutils
7) install scipy 

### Algorithm
● We utilised a pre trained frontal face detector from Dlib’s library which is based on  a modification to the Histogram of Oriented Gradients in combination with Linear  SVM for classification.  

● The pre-trained facial landmark detector inside the dlib library is used to estimate  the location of 68 (x, y)-coordinates that map to facial structures on the face. The 68  landmark output is shown in the figure below

<img src="https://github.com/jaisayush/Fatigue-Detection-System-Based-On-Behavioural-Characteristics-Of-Driver/blob/master/face.PNG">

● We then calculate the aspect ratio to check whether eyes are opened or closed.

● The eye is open if Eye Aspect ratio is greater than threshold. (Around 0.3)

<img src="https://github.com/jaisayush/Fatigue-Detection-System-Based-On-Behavioural-Characteristics-Of-Driver/blob/master/eye.PNG">

● A blink is supposed to last 200-300 milliseconds.

### Running
```
activate <env>
```
```
python detec_drowsiness.py -p shape_predictor_68_face_landmarks.dat -a alarm.wav
```



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details