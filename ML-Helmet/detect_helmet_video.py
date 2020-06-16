# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
# from sleep_detec import sleep_detec, eye_aspect_ratio
# import playsound
from pygame import mixer

mixer.init()
ok = mixer.Sound('ok.wav')
notok = mixer.Sound('notok.wav')


def detect_and_predict_mask(frame, faceNet, chinNet, helmetNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds1 = []
    preds2 = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY-100))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            chin = frame[int(startY+250):endY+50,
                         int(startX-startX*1/3):endX+50]
            chin = cv2.cvtColor(chin, cv2.COLOR_BGR2RGB)
            chin = cv2.resize(chin, (224, 224))
            chin = img_to_array(chin)
            chin = preprocess_input(chin)
            chin = np.expand_dims(chin, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            # locs.append((startX, startY, endX, endY))
            locs.append((int(startX-startX*1/3),
                         int(startY+250), endX+50, endY+50))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds1 = chinNet.predict(chin)
        preds2 = helmetNet.predict(faces)
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds1, preds2)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m1", "--model1", type=str,
                default="chin_detector.model",
                help="path to trained hat straps detector model 1")
ap.add_argument("-m2", "--model2", type=str,
                default="helmet_detector.model",
                help="path to trained helmet detector model 2")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the hat straps detector model from disk
print("[INFO] loading face mask detector model...")
chinNet = load_model(args["model1"])

# load the helmet detector model from disk
print("[INFO] loading face helmet detector model...")
helmetNet = load_model(args["model2"])


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

counter = 0
ALARM_ON = True
# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    (locs, preds1, preds2) = detect_and_predict_mask(
        frame, faceNet, chinNet, helmetNet)

    for (box, pred1, pred2) in zip(locs, preds1, preds2):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (straps, withoutTraps) = pred1
        (helmet, withoutHelmet) = pred2
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label1 = "Hat straps" if straps < withoutTraps else "No hat straps"
        label2 = "Helmet" if helmet < withoutHelmet else "No helmet"
        color = (0, 255, 0) if label1 == "Hat straps" and label2 == "Helmet" else (
            0, 0, 255)
        # include the probability in the label
        label1 = "{}: {:.2f}%".format(label1, max(straps, withoutTraps) * 100)
        label2 = "{}: {:.2f}%".format(label2, max(helmet, withoutHelmet) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label1, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, label2, (startX, startY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if color == (0, 255, 0):
            counter += 1
            print(counter)
            if counter >= 10:
                if not ALARM_ON:
                    ALARM_ON = True
                    cv2.putText(frame, "tat ca OK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
                    # playsound.playsound("ok.wav")
                    ok.play()
                counter = 0
        else:
            counter -= 1
            print(counter)
            if counter <= -10:
                if ALARM_ON:
                    ALARM_ON = False
                    # playsound.playsound("notok.wav")
                    notok.play()
                counter = 0
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
