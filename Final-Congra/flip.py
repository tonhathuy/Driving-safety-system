import os
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=False,
                help="path to output dataset face")
args = vars(ap.parse_args())
parent_dir = args["dataset"]
directory = args["output"]
print(parent_dir)
print(directory)
path = os.path.join(parent_dir, directory)
if not os.path.exists(path):
    os.mkdir(path)
print(path)
imagePaths = list(paths.list_images(args["dataset"]))
for imagePath in imagePaths:
    print(imagePath)
    img = cv2.imread(imagePath)
    img_flip = cv2.flip(img, 1)
    path_splits = os.path.splitext(imagePath)
    cv2.imwrite(path_splits[0] + '_flip.jpg', img_flip)
