import cv2
import os
from mask import create_mask
from imutils import paths
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))

# folder_path = "without_mask/"
# #dist_path = "/home/preeth/Downloads"

# #c = 0
# images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# for i in range(len(images)):
#     print("the path of the image is", images[i])
#     #image = cv2.imread(images[i])
#     #c = c + 1
#     create_mask(images[i])



#c = 0
# images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for imagePath in imagePaths:
    # print("the path of the image is", images[i])
    # #image = cv2.imread(images[i])
    # #c = c + 1
    # create_mask(images[i])
    # print(imagePath)
    create_mask(imagePath)