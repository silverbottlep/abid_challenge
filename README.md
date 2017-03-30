# Amazon Bin Image Dataset(ABID) Challenge

The Amazon Bin Image Dataset contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. You can download and find the details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/).  We propose 3 different tasks, which would be practically useful when we want to double-check the bins before they are packed or delivered. Tasks are sometimes very challenging because of heavy occlusions and a large number of object categories. We would like to open a new challenge in order to attract talented researchers in both academia and industries for these tasks. As a starting point, we provide baseline methods and pre-trained models for two tasks, counting and object verification tasks.

## 1. Tasks

### 1.1. Counting
This is a simple task that you are supposed to count every object instances in the bin. This is object category agnostic task, which means if there are two same objects in the bin, you count them as two.
![counting](http://www.cs.unc.edu/~eunbyung/abidc/counting.png)

### 1.2. Object verification 
This is a task for verifying the presence of the object in the bin. You will be given an image and question pair. The question contains object category and presence, e.g. 'Is there a toothbrush in the bin?'. your program should be able to give an answer 'yes' or 'no'.  
![obj_verification](http://www.cs.unc.edu/~eunbyung/abidc/obj_verification.png)

### 1.3. Object quantity verification 
This is a task for verifying the quantity of the object in the bin. You will be given an image and question pair. The question contains the quantity of the object, e.g. 'are there 2 toothbrush in the bin?', your program should be able to give an answer 'yes' or 'no'.

![obj_quant_verification](http://www.cs.unc.edu/~eunbyung/abidc/obj_quant_verification.png)

## 2. Dataset

These are some typical images in the dataset. A bin contains multiple object categories and various number of instances. The corresponding metadata exist for each bin image and it includes the object category identification(Amazon Standard Identification Number, ASIN), quantity, size of objects, weights, and so on. The size of bins are various depending on the size of objects in it. The tapes in front of the bins are for preventing the items from falling out of the bins and sometimes it might make the objects unclear. Objects are sometimes heavily occluded by other objects or limited viewpoint of the images.

![abid_images](http://www.cs.unc.edu/~eunbyung/abidc/abid_images.png)

### 2.1 Metadata

![ex1](http://www.cs.unc.edu/~eunbyung/abidc/image1_small.jpg)

```
{
    "BIN_FCSKU_DATA": {
        "B00CFQWRPS": {
            "asin": "B00CFQWRPS",
            "height": {
                "unit": "IN",
                "value": 2.399999997552
            },
            "length": {
                "unit": "IN",
                "value": 8.199999991636
            },
            "name": "Fleet Saline Enema, 7.8 Ounce (Pack of 3)",
            "normalizedName": "(Pack of 3) Fleet Saline Enema, 7.8 Ounce",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 1.8999999999999997
            },
            "width": {
                "unit": "IN",
                "value": 7.199999992656
            }
        },
        "ZZXI0WUSIB": {
            "asin": "B00T0BUKW8",
            "height": {
                "unit": "IN",
                "value": 3.99999999592
            },
            "length": {
                "unit": "IN",
                "value": 7.899999991942001
            },
            "name": "Kirkland Signature Premium Chunk Chicken Breast Packed in Water, 12.5 Ounce, 6 Count",
            "normalizedName": "Kirkland Signature Premium Chunk Chicken Breast Packed in Water, 12.5 Ounce, 6 Count",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 5.7
            },
            "width": {
                "unit": "IN",
                "value": 6.49999999337
            }
        },
        "ZZXVVS669V": {
            "asin": "B00C3WXJHY",
            "height": {
                "unit": "IN",
                "value": 4.330708657
            },
            "length": {
                "unit": "IN",
                "value": 11.1417322721
            },
            "name": "Play-Doh Sweet Shoppe Ice Cream Sundae Cart Playset",
            "normalizedName": "Play-Doh Sweet Shoppe Ice Cream Sundae Cart Playset",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 1.4109440759087915
            },
            "width": {
                "unit": "IN",
                "value": 9.448818888
            }
        }
    },
    "EXPECTED_QUANTITY": 3
}
```

This is an example of image(jpg) and metadata(json) pair. This image contains 3 different object categories. For each category, there is one instance. So, "EXPECTED_QUANTITY" is 3, and for each object category "quantity" field was 1. Unique identifier("asin") is assigned to each object category, e.g. here "B00CFQWRPS", "B00T0BUKW8", and "B00C3WXJHY". 

### 2.2 Dataset statistics
| Description | Quantity |
|----------------------|--------|
| The number of images | 535,050 |
| Average quantity in a bin | 5.1 |
| The number of object categories | 459,476 |

![stats](http://www.cs.unc.edu/~eunbyung/abidc/stats.png)

The left figure shows the distribution of quantity in a bin(90% of bin images contains less then 10 object instances in a bin). The right figure shows the distribution of object repetition. 164,255 object categories (out of 459,475) showed up only once across entire dataset, and 164,356 object categories showed up twice. The number of object categories that showed up 10 times was 3038.

## 3. Data preparation

### 3.0 Prerequisite
1. [PyTorch](https://github.com/pytorch/pytorch)
2. [torch-vision](https://github.com/pytorch/vision)
3. This code is heavily based on pytorch [example codes](https://github.com/pytorch/examples)

### 3.1 Downloading data
You need to download the dataset first and locate images and metadata in same directory, e.g. $(data)/public_images, $(data)/metadata. Soft link to them in dataset directory. For downloading, find more details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/)
```
$(abid_challenge_root)/dataset> ln -s $(data) ./data
```

### 3.2 Training and validation split
You can make your own training/validation split. For example, we provide code to randomly split, which will generate 'random_train.txt' and 'random_val.txt' files.
```
$(abid_challenge_root)/dataset> python random_split.py
```

We also provide train/val splits that we used to train our baseline methods and you can download it by running
```
$(abid_challenge_root)/dataset> ./get_random_split.sh
```

Next, we will make metadata.
```
$(abid_challenge_root)/dataset> python make_metadata.py
```
You will see 'metadata.json'(~640M) and 'instances.json'(76M) files. 'metadata.json' file contains all metadata in a single file, and 'instances.json' file contains a list of all object categories and image indices that contain the object. They will be useful for investigating metadata of the datasets and used to generate task specific metadata files. For your convinience, we also provide pre-computed files, and you can download it by
```
$(abid_challenge_root)/dataset> ./get_metadata.sh
```

### 3.3 Task specific metadata
Once you have train/val split and metadata files, now you can generate task specific metadata files. This will be used when you train the baseline methods.
```
$(abid_challenge_root)/dataset> python make_counting_data.py
$(abid_challenge_root)/dataset> python make_obj_verification_data.py
```
This will generate 'counting_train.json', 'counting_val.json', 'obj_verification_train.json', 'obj_verification_val.json' files. Alternatively, You can simply download pre-processed verification data for training and validataion sets, which we used for making the baseline methods.
```
$(abid_challenge_root)/dataset> ./get_task_data.sh
```

### 3.4 Resizing images
For baseline methods, we resized all image into 224x224 for convinient training purpose. You will have new directory $(data)/public_images_resize that contain resized images
```
$(abid_challenge_root)/dataset> python resize_image.py
```

### 3.5 Moderate and hard task
We divide the each task into two levels of difficulty(moderate and hard). For moderate difficulty, you will be tested over the bin images that contain upto 5 objects. For hard task, you will be tested over all bin images. You can submit your results whatever you are interseted (both, or one of them). As baseline methods, we provide ones for moderate difficulty.

## 4. Deep Convolutional Classification Network for Counting
It is a simple classification network for counting task. The deep CNN will classify the image as one of 6 categories(0-5, for moderate difficulty). We used resnet 34 layer architecture and trained from the scratch. 

### 4.1 Training
```
$(abid_challenge_root)/counting> mkdir snapshots
$(abid_challenge_root)/counting> CUDA_VISIBLE_DEVICES=0 python train.py ~/Works/data/amazon_bin/public_images_resize/ -a resnet34 --epochs 40 --lrd 10
```
It will run 40 epochs, and every 10 epochs learning rate will decay by a factor of 0.1. One epoch means the network goes through all training images once. Batch size is 128. Following shows loss curves and validation accuracy. Here we got best validation accuracy at 21 epoch. As you might notice, it will start to overfit after 21 epoch.

![train_loss](http://www.cs.unc.edu/~eunbyung/abidc/counting_train_loss.png)
![val_acc](http://www.cs.unc.edu/~eunbyung/abidc/counting_val_acc.png)

#### 4.1.1 Pretrain models
You can download pre-trained models [here](http://www.cs.unc.edu/~eunbyung/abidc/resnet34_best.pth.tar)

### 4.2 Evaluation on validataion sets
```
$(abid_challenge_root)/verification_siamese> CUDA_VISIBLE_DEVICES=0 python train.py ~/Works/data/amazon_bin/public_images_resize/ -a resnet34 --evaluate True --resume ./snapshots/resnet34_best.pth.tar
```
You should be able to get 57.4% accuracy. You also get output file 'counting_result.txt', which is going to be your submission file format. Each line contains a integer value(count) corresponding to one image being evaluated. You will be evaluated by two metrics, accuracy and RMSE(Root Mean Square Error). Following shows the results on validation split.

| Accuracy(%) | RMSE(Root Mean Square Error)|
|----------------------|--------|
| 57.42 | 0.886 |

| Quantity | Per class accuracy(%) | Per class RMSE |
|-------|--------------------|----------------|
| 0 | 97.7 | 0.207 |
| 1 | 83.4 | 0.569 |
| 2 | 67.2 | 0.728 |
| 3 | 54.9 | 0.810 |
| 4 | 42.6 | 0.949 |
| 5 | 44.9 | 1.245 |

## 5. Deep Convolutional Siamese Network for Object Verification
We provide one baseline method for object verification task. Since the number of object categories are huge, we propose to learn how to compare the images instead of modelling all individual object categories. For example, we pick positive pair, which both images should contain at least one common object(no common object for negative pair), and train the network to classify the pair as positive or negative. Siamese network would be proper architectural choice for this purpose.

This is based on following paper, Siamese neural networks for one-shot image recognition, Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, ICML Deep Learning Workshop, 2015, [pdf](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). 

![siamese](http://www.cs.unc.edu/~eunbyung/abidc/siamese_small.png)

When testing, we are given one image and the name of object category(asin) in question. From training images, we pick all images that contain the object category in question, and make pairs with the given image. And, we will make final decision as a majority votes based on the results from all pairs. More formally,

![equation](http://www.cs.unc.edu/~eunbyung/abidc/eq.png)

,where S is the set of {x1,x2} image pair, one is test image and one from training images that contains the object category in question. 1[ ] is an indicator function and p is estimated probability of being positive by siamese network.


### 5.1 Training
```
$(abid_challenge_root)/verification_siamese> mkdir snapshots
$(abid_challenge_root)/verification_siamese> CUDA_VISIBLE_DEVICES=0 train.py $(data)/public_images_resize/ -a resnet34 -b 128 --epochs 40
```
It will run 40 epochs, and every 10 epochs learning rate will decay by a factor of 0.1. 1 epoch means the network goes through all training images once. Resnet34 architecture will be used and it will be trained from the scratch and batch size is 128. Following shows loss curves and validation accuracy. Here we got best validation accuracy at 36 epoch.

![train_loss](http://www.cs.unc.edu/~eunbyung/abidc/train_loss.png)
![val_acc](http://www.cs.unc.edu/~eunbyung/abidc/val_acc.png)

#### 5.1.1 Pretrain models
You can download pre-trained models [here](http://www.cs.unc.edu/~eunbyung/abidc/resnet34_siamese_best.pth.tar)

### 5.2 Evaluation on validataion sets
```
$(abid_challenge_root)/verification_siamese> CUDA_VISIBLE_DEVICES=0 python train.py ~/Works/data/amazon_bin/public_images_resize/ -a resnet34 --evaluate True --resume ./snapshots/resnet34_siamese_best.pth.tar
```
You should be able to get 76.8% accuracy. You also get output file 'obj_verification_result.txt', which is going to be your submission file format. Each line contains a binary value(0 or 1) corresponding to one pair(image and question). You will be evaluated by accuracy metric.

Followings are a few of examples. You are given a question, e.g. Is there an Prebles' Artforms(11th Edition) in the bin? and an image in question(left image). The right image is picked from training image that the contains the object in question. Pred is prediction of the network and GT is ground truth. We also put estimated probability in the bracket. If GT is 'yes', then both images should have the object in question. If GT is 'no', then both images should not have the object in question.

![example1](http://www.cs.unc.edu/~eunbyung/abidc/example1_.png)

![example2](http://www.cs.unc.edu/~eunbyung/abidc/example2_.png)

![example3](http://www.cs.unc.edu/~eunbyung/abidc/example3_.png)

![example4](http://www.cs.unc.edu/~eunbyung/abidc/example4_.png)

![example5](http://www.cs.unc.edu/~eunbyung/abidc/example5_.png)

![example6](http://www.cs.unc.edu/~eunbyung/abidc/example6_.png)

![example7](http://www.cs.unc.edu/~eunbyung/abidc/example7_.png)
