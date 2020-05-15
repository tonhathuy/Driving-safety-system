# Helmet Detector
Create a light traffic detector using OpenCV, TensorFlow API, and Deep Learning.
## Requirements
 - Tensorflow 1.14 
 
## Installation
follow the tutorial how to setup [Tensorflow for Object Detection](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) 

/home/huy/miniconda3/models/research

## Training 
### 1. Choosing a model - download, compress and copy to /models/research/object_detection
[download Tensorflow models from model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
- [ssd_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
- [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
### 2. clone github
- clone and copy all folder and file above to /models/research/object_detection
### 3. run training
```bash
python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/ssd_inception_v2_pets.config
```
About every 5 minutes the current loss gets logged to Tensorboard. We can open Tensorboard by opening a second command line
```bash
tensorboard --logdir=training
```
### 4. Exporting inference graph
we need to navigate to the training directory and look for the model.ckpt file with the biggest index
Then we can create the inference graph by typing the following command in the command line:
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
### 5. Testing object detector
run file object_detection_with_own_model.ipynb with jupyter notebook



