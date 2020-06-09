import cv2
import numpy as np

import os
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
    image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    cropped = image[int(height*2/3):height-20, int(width/3):int(width*2/3)]
    path_splits = os.path.splitext(imagePath)
    test = path_splits[0].split("/")
    print(test[0])
    print(test[1])
    print(path)
    path_cut_file = path+"/"+test[1]+"cut"+".jpg"
    try:
        cv2.imwrite(path_cut_file,cropped)
    except:
        print("error image")
        pass
  