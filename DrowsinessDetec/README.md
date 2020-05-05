# DROWNSINESS DETECTION

This program is used to detect drowsiness for any given person. In this program we check how long a person's eyes have been closed for. If the eyes have been closed for a long period, the program will alert the user by playing an alarm sound.

### Prerequisites

1) install python
2) install pip
3) install opencv
4) install cmake
-  install dlib: 
   -
   ```
   pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
   ```
5) install playsound
6) install imutils
7) install scipy 

### Running
```
activate <env>
```
```
python detec_drowsiness.py -p shape_predictor_68_face_landmarks.dat -a alarm.wav
```



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details