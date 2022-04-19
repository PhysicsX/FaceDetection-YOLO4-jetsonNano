# Face Detection example with YOLO4 on Jetson Nano

This example is tested with ubuntu
```bash
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.6 LTS
Release:        18.04
Codename:       bionic
```
Tested jetpack version
```bash
ulas@ulas:~$ sudo apt-cache show nvidia-jetpack
[sudo] password for ulas:
Package: nvidia-jetpack
Version: 4.6-b199
Architecture: arm64
Maintainer: NVIDIA Corporation
Installed-Size: 194
Depends: nvidia-cuda (= 4.6-b199), nvidia-opencv (= 4.6-b199), nvidia-cudnn8 (= 4.6-b199), nvidia-tensorrt (= 4.6-b199), nvidia-visionworks (= 4.6-b199), nvidia-container (= 4.6-b199), nvidia-vpi (= 4.6-b199), nvidia-l4t-jetson-multimedia-api (>> 32.6-0), nvidia-l4t-jetson-multimedia-api (<< 32.7-0)
Homepage: http://developer.nvidia.com/jetson
Priority: standard
Section: metapackages
Filename: pool/main/n/nvidia-jetpack/nvidia-jetpack_4.6-b199_arm64.deb
Size: 29368
SHA256: 69df11e22e2c8406fe281fe6fc27c7d40a13ed668e508a592a6785d40ea71669
SHA1: 5c678b8762acc54f85b4334f92d9bb084858907a
MD5sum: 1b96cd72f2a434e887f98912061d8cfb
Description: NVIDIA Jetpack Meta Package
Description-md5: ad1462289bdbc54909ae109d1d32c0a8

Package: nvidia-jetpack
Version: 4.6-b197
Architecture: arm64
Maintainer: NVIDIA Corporation
Installed-Size: 194
Depends: nvidia-cuda (= 4.6-b197), nvidia-opencv (= 4.6-b197), nvidia-cudnn8 (= 4.6-b197), nvidia-tensorrt (= 4.6-b197), nvidia-visionworks (= 4.6-b197), nvidia-container (= 4.6-b197), nvidia-vpi (= 4.6-b197), nvidia-l4t-jetson-multimedia-api (>> 32.6-0), nvidia-l4t-jetson-multimedia-api (<< 32.7-0)
Homepage: http://developer.nvidia.com/jetson
Priority: standard
Section: metapackages
Filename: pool/main/n/nvidia-jetpack/nvidia-jetpack_4.6-b197_arm64.deb
Size: 29356
SHA256: 104cd0c1efefe5865753ec9b0b148a534ffdcc9bae525637c7532b309ed44aa0
SHA1: 8cca8b9ebb21feafbbd20c2984bd9b329a202624
MD5sum: 463d4303429f163b97207827965e8fe0
Description: NVIDIA Jetpack Meta Package
Description-md5: ad1462289bdbc54909ae109d1d32c0a8
```

Darknet commit number:
```bash
ulas@ulas:~/darknet$ git log --oneline
8a0bf84 (HEAD -> master, origin/master, origin/HEAD) various fixes (#8398)
2c137d1 Fix conv_lstm cuda errors and nan's (#8388)
b4d03f8 issue #8308: memory leaks in map (#8314)
```

First Darkent should be compiled on the Jetson Nano. Shared objectfile is available in the repo.
If you do not want to compile use exactly same versions as described here. Otherwise compile yourself.
```bash
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
git clone https://github.com/AlexeyAB/darknet
```
Some parameters should be updated according to used hardware which is nano. You can use the Makefile in this repository as a reference.
Update the Makefile. 
```bash
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]
```


Then compile the darknet.
```bash
make
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```
You can test it with:
You should be in the darknet directory to run these examples:
```bash
./darknet detector demo cfg/coco.data cfg/yolov4-csp.cfg yolov4-csp.weights data/dog.jpg -gpus 0
```
for usb camera:
```bash
./darknet detector demo cfg/coco.data cfg/yolov4-csp.cfg yolov4-tiny.weights -c 1 -gpus 0
```
for gstreamer ( should be builded or installed )
```bash
./darknet detector demo cfg/coco.data cfg/yolov4-csp.cfg yolov4-csp.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=800, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2  ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink" -gpus 0
``` 

I compiled opencv myself. The version is 4.4.0. If you want to build yourself, use the script to build the opencv. It will delete the old ones. Script is tested on nano only.

## Train the model
Before starting the training I created a directory inside the darknet directoy. That means binary is located in one above directory. It does not matter as long as you are able to give the correct paths for images and related configuration files then you can run the binary from wherever you want.
```bash
mkdir trainingYolov4 && cd trainingYolov4
```

I used wider face dataset to train the model. Also transfer learning is used.
Install the zip files from http://shuoyang1213.me/WIDERFACE/
unzip them. There are three zip files.
First one is for training. Second one is for validation for the images.
Last one has the txt files which holds the related position data. But these data should be in the same file with images and with same name. We will handle this later with single python script.

```bash
unzip ../../drive/MyDrive/WIDER_train.zip
unzip ../../drive/MyDrive/WIDER_val.zip
unzip ../../drive/MyDrive/wider_face_split.zip
```
After this lets create txt file for each picture for yolo.
Before start edit the file for absolute paths
In the code:
```bash
train_set_dir = '/content/drive/MyDrive/trainingFace/darknet/trainingYolo/WIDER_train/images'
val_set_dir = '/content/drive/MyDrive/trainingFace/darknet/trainingYolo/WIDER_val/images'

train_gt = '/content/drive/MyDrive/trainingFace/darknet/trainingYolo/wider_face_split/wider_face_train_bbx_gt.txt'
val_gt = '/content/drive/MyDrive/trainingFace/darknet/trainingYolo/wider_face_split/wider_face_val_bbx_gt.txt
```

```bash
wget https://raw.githubusercontent.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/master/make_annotation.py
python3 make_annotation.py
```
After this check the each directory for images you will see txt files for each image. Additionaly this script will produce two txt files where you run it. train.txt and val.txt. These txt files have all pictures absolute paths.
We will need it in the obj.data file (which is the yolo).

Get related configration files
```bash
wget https://raw.githubusercontent.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/master/make_annotation.py
#check the related part in the darknet repo for this
wget https://raw.githubusercontent.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/master/yolo/yolov4-tiny-3l.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29
wget https://raw.githubusercontent.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/master/yolo/obj.names
wget https://raw.githubusercontent.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/master/yolo/obj.data
```
yolov4-tiny.conv.2 is the training model that we can use it for transfer learning even if it was not trained for our desired object.

create a yolo directory and put obj.data, obj.names and cfg file to this directory
```bash
mkdir yolo
mv -t yolo obj.data obj.names yolov4-tiny-3l.cfg
```
In obj.data update the paths for train.txt and val.txt which is produced by the annotayion.py file.
```bash
train = /content/drive/MyDrive/trainingFace/darknet/trainingYolo/train.txt
valid = /content/drive/MyDrive/trainingFace/darknet/trainingYolo/val.txt
```
If you do not run the darknet binary under the darknet directory, during the traning it will look for bakcup directory. So lets create here one.
otherwise training will fail
```bash
mkdir backup
```
Start training with
```bash
./../darknet detector train yolo/obj.data yolo/yolov4-tiny-3l.cfg yolov4-tiny.conv.29 backup\yolov4-tiny-3l_last.weights -map -dont_show
```

If you run these command on colab then be sure GPU is active
```bash
!nvidia-smi
```

You can check the process from chart.png in the same directory

![](https://github.com/PhysicsX/FaceDetection-YOLO4-jetsonNano/chart_yolov4-tiny-3l.png?raw=true)

## Run the example with python
If examples are working. At least the one for picture. Then we can use python wrappers to call darknet.so file and use desired functions inside the python. 
Here it will be good to use opencv with python. Everything will be more easy.

just run the application with python3.
```bash
python3 face_detection.py
```


