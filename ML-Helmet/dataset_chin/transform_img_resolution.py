from PIL import Image
import os 
import argparse

def rescale_imgs(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+"resized/"+img)

def name():
    print('hello')
    parser = argparse.ArgumentParser(description="Rescale image")
    parser.add_argument('-d', '--directory', type=str, help='diectory img', required=True)
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True)
    args = parser.parse_args()
    rescale_imgs(args.directory, args.size)

name()