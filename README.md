# Amazon Bin Image Dataset(ABID) Challenge
This includes codes of parsing and preprocessing [ABID dataset](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/) and a baseline method for a bin verification task for ABID challenge. It will be helpful for challenge participants and someone doing research with ABID datasets in general. As a tester challenge of [ILSVRC2017](http://image-net.org/challenges/LSVRC/2017/), and algorithmic details of leading entries will be presented at [Beyond ILSVRC workshop](http://image-net.org/challenges/beyond_ilsvrc.php) in conjunction with [CVPR 2017](http://cvpr2017.thecvf.com/).

## 0. Downloading dataset
Find the details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/)

## 0. ShapeNet dataset download
You should have ShapeNetCore.v1 dataset in your local $(SHAPENET_DATA) directory via [shapenet.org](https://shapenet.org/) in your local directory. We will use entire models for car category. For chair category, we used train/test split suggested by appearance flow network paper[[link]](https://github.com/tinghuiz/appearance-flow)(They picked the models that have rich textures).
```bash
$(tvsn_root)/tvsn/data$>./make_new_chair.sh $(SHAPENET_DATA)
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/02958343 ./car
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/new_chair ./chair
```

## 1. Dataset Preparation (Rendering multiple view images)
I adopted [rendering engine](https://github.com/sunweilun/ObjRenderer) used in the [appearance flow network](https://github.com/tinghuiz/appearance-flow), and modified original code a little bit to get the surface normals and object coordinates, which will be used for generating visibility maps. You can download from [here](https://github.com/silverbottlep/ObjRenderer) and edit the 'config.txt' file to tune the engine for your purpose. For example,
```bash
$(tvsn_root)$> git clone git@github.com:silverbottlep/ObjRenderer.git
$(tvsn_root)$> cd ObjRenderer
$(tvsn_root)/ObjRenderer$> cat config.txt
folder_path = $(SHAPENET_DATA)/02958343 "e.g. 'car' category"
envmap_path = envmaps/envmap2.hdr
theta_inc = 20
phi_inc = 10
phi_max = 20
output_coord = 1
output_norm = 1
render_size = 1024
output_size = 256
reverse_normals = 1
brightness = 0.7
```
Build and execute. It will take long time, and requires a lot of space(~10GB)
```bash
$(tvsn_root)/ObjRenderer$> make
$(tvsn_root)/ObjRenderer$> ./dist/Release/GNU-Linux-x86/objrenderer
```

## 2. Dataset Preparation (Generating visibility maps)
Now, we are going to make visibility maps.  For convinience, we provide precomputed visibilty maps. You can download them from following links, and locate them in $(tvsn_root)/tvsn/data directory

[maps_car.t7](https://drive.google.com/open?id=0B-r7apOz1BHAVEI1RURZYUl4Tlk) (~26G)

[maps_chair.t7](https://drive.google.com/open?id=0B-r7apOz1BHANGlsY1k3Z29yVEU) (~2G)

## 5. Downloading pretrained models
We provide pretrained models for car and chair category. You can download it from following links.

[tvsn_car_epoch220.t7](https://drive.google.com/open?id=0B-r7apOz1BHAQVVXR0JXcTh5MUk) (~134M)

[tvsn_chair_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAWmQtdEZ6ZG5udW8) (~134M)

[doafn_car_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAR1RKWXM1c1NBekk) (~351M)

[doafn_chair_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAaWh4N1Vnc3hKdE0)(~351M)
