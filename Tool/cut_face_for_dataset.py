import face_recognition
import cv2
import numpy as np

import os
from mask import create_mask
from imutils import paths
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output", required=False, default="cut",
	help="path to output dataset face")
args = vars(ap.parse_args())
parent_dir = args["dataset"]
directory = args["output"]
path = os.path.join(parent_dir, directory)
if not os.path.exists(path):
    os.mkdir(path)
print(path) 
imagePaths = list(paths.list_images(args["dataset"]))
for imagePath in imagePaths:
    print(imagePath)
        
    # image = face_recognition.load_image_file("helmet/000102.jpg")
    image = face_recognition.load_image_file(imagePath)
    face_locations = face_recognition.face_locations(image)
    image = cv2.imread(imagePath)
    # print(face_locations)
    for i in range(len(face_locations)):
        face_location = face_locations[i]
        print(face_location)

        # img = cv2.imread("lenna.png")
        # crop_img = img[y:y+h, x:x+w]
        # cv2.imshow("cropped", crop_img)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(image.shape)
        height, width = image.shape[:2]
        # Each location contains positions in order: top, right, bottom, left
        addhead = ((face_location[2] - face_location[0])*3/5)	
        addhead = int(addhead)
        print("addhead: ",addhead)
        headafter = face_location[0]-addhead  if (face_location[0]-addhead) > 0 else 0
        print(headafter)
        top_left = (face_location[3], headafter)
        bottom_right = (face_location[1], face_location[2])

        # Get color by name using our fancy function
        # color = name_to_color(match)

        # Paint frame
        # cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
        
        addwidth = (face_location[1]-face_location[3])/4
        addwidth = int(addwidth)


        cropped = image[headafter:face_location[2]+int(addhead/2), face_location[3]-addwidth:face_location[1]+addwidth]
        path_splits = os.path.splitext(imagePath)
        test = path_splits[0].split("/")
        print(test[0])
        print(test[1])
        print(path)
        path_cut_file = path+"/"+test[1]+"cut"+str(i)+".jpg"
        try:
            cv2.imwrite(path_cut_file,cropped)
        except:
            print("error image")
            pass
        
