# Amazon Bin Image Dataset(ABID) Challenge

## 1. Amazon Bin Image Dataset
The Amazon Bin Image Dataset contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. You can download and find the details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/).

![abid_images](http://www.cs.unc.edu/~eunbyung/abidc/abid_images.png)

These are some typical images in the dataset. A bin contains multiple object categories and various number of instances. The corresponding metadata exist for each bin image and it includes the object category identification(Amazon Standard Identification Number, ASIN), quantity, size of objects, weights, and so on. The size of bins are various depending on the size of objects in it. The tapes in front of the bins are for preventing the items from falling out of the bins and sometimes it might make the objects unclear. Objects are sometimes heavily occluded by other objects or limited viewpoint of the images.

### 1.1 Metadata
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

### 1.2 Dataset statistics
| Description | Quantity |
|----------------------|--------|
| The number of images | 535,050 |
| Average quantity in a bin | 5.1 |
| The number of object categories | 459,476 |

![stats](http://www.cs.unc.edu/~eunbyung/abidc/stats.png)

The left figure shows the distribution of quantity in a bin(90% of bin images contains less then 10 object instances in a bin). The right figure shows the distribution of object repetition. 164,255 object categories (out of 459,475) showed up only once across entire dataset, and 164,356 object categories showed up twice. The number of object categories that showed up 10 times was 3038.

## 2. Bin Verification Task
We propose bin verification task with this dataset. Given an image and question pair, e.g. 'Is there a toothbrush in the bin?', your program should be able to give an answer 'yes' or 'no'. Although it looks simple binary question, it is very challenging task because of a large number of object categories and limited views(occlusion) of the objects. This task would be practically useful because we might want to double-check the presence of the objects before they are packed or delivered. 

![verification](http://www.cs.unc.edu/~eunbyung/abidc/verification.png)

## 3. Deep Convolutional Siamese Network for Bin Verification
We provide one baseline method for bin verification task. Since the number of object categories are huge, we propose to learn how to compare the images instead of modelling all individual object categories. For example, we pick positive pair, which both images should contain at least one common object(no common object for negative pair), and train the network to classify the pair as positive or negative. Siamese network would be proper architectural choice for this purpose.

This is based on following paper, Siamese neural networks for one-shot image recognition, Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, ICML Deep Learning Workshop, 2015, [pdf](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). 

![siamese](http://www.cs.unc.edu/~eunbyung/abidc/siamese_small.png)

When testing, we are given one image and the name of object category(asin) in question. From training images, we pick all images that contain the object category in question, and make pairs with the given image. And, we will make final decision as a majority votes based on the results from all pairs. More formally,

![equation](http://www.cs.unc.edu/~eunbyung/abidc/eq.png)

,where S is the set of {x1,x2} image pair, one is test image and one from training images that contains the object category in question. 1[ ] is an indicator function and p is estimated probability of being positive by siamese network.

### 3.0 Prerequisite
1. [PyTorch](https://github.com/pytorch/pytorch)
2. [torch-vision](https://github.com/pytorch/vision)
3. This code is heavily based on pytorch [example codes](https://github.com/pytorch/examples)

### 3.1 Data Preparation
You need to download the dataset first and locate images and metadata in same directory, e.g. $(data)/public_images, $(data)/metadata. Soft link to them in dataset directory. For downloading, find more details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/)
```
$(abid_challenge_root)/dataset> ln -s $(data) ./data
```
Now, we are going to make verification data which will be used for training. For training purpose, we only consider the images that contain less than 6 quantity in a bin. If there are a large number of objects in a bin, sometimes it is hard to see the objects since they are heavily occluded each other.

#### 3.1.1 Downloading pre-processed verification data
You can simply download pre-processed verification data for training and validataion sets by running script. You will get 'verification_train.json' and 'verification_val.json' files in $(abid_challenge_root)/dataset directory 
```
$(abid_challenge_root)/dataset> ./get_verification_data.sh
```

#### 3.1.2 (optional) Generating verification data
You can optionally make your own verification data files. First, let's randomly split images into train and validataion sets. By running script, you will get 'random_train.txt' and 'random_val.txt' files.
```
$(abid_challenge_root)/dataset> python random_split.py
```

Next, we will make metadata before we generate verification data. It can be also easily done by
```
$(abid_challenge_root)/dataset> python make_metadata.py
```

You will see 'metadata.json'(~640M) and 'instances.json'(76M) files. 'metadata.json' file contains all metadata in a single file, and 'instances.json' file contains a list of all object categories and image indices that contain the object. They will be used to generate verification data. Let's run another script to generate 'verification_train.json' and 'verification_val.json' files.
```
$(abid_challenge_root)/dataset> python make_verification_data.py
```

#### 3.1.3 Resizing images
For convinient training purpose, we will resize all image into 224x224. You will have new directory $(data)/public_images_resize that contain resized images
```
$(abid_challenge_root)/dataset> python resize_image.py
```

### 3.2 Training
Now, it's time to train!
```
$(abid_challenge_root)/verification_siamese> mkdir snapshots
$(abid_challenge_root)/verification_siamese> CUDA_VISIBLE_DEVICES=0 train.py $(data)/public_images_resize/ -a resnet34 -b 128 --epochs 40
```
It will run 40 epochs, and every 10 epochs learning rate will decay by a factor of 0.1. 1 epoch means the network goes through all training images once. Resnet34 architecture will be used and it will be trained from the scratch and batch size is 128. Following shows loss curves and validation accuracy. Here we got best validation accuracy at 36 epoch.

![train_loss](http://www.cs.unc.edu/~eunbyung/abidc/train_loss.png)
![val_acc](http://www.cs.unc.edu/~eunbyung/abidc/val_acc.png)

#### Pretrain models
You can download pre-trained models [here](http://www.cs.unc.edu/~eunbyung/abidc/resnet34_siamese_best.pth.tar)

### 3.3 Evaluation on validataion sets
```
$(abid_challenge_root)/verification_siamese> CUDA_VISIBLE_DEVICES=0 python train.py ~/Works/data/amazon_bin/public_images_resize/ -a resnet34 --evaluate True --resume ./snapshots/resnet34_siamese_best.pth.tar
```

Followings are a few of examples. The left image is the image in question, and the right image is from training image. Pred is prediction of the network and GT is ground truth. We also put estimated probability in the bracket. If GT is 1, then both images should have common object. If GT is 0, then both images should not have common object.

![example1](http://www.cs.unc.edu/~eunbyung/abidc/example1.png)

![example2](http://www.cs.unc.edu/~eunbyung/abidc/example2.png)

![example3](http://www.cs.unc.edu/~eunbyung/abidc/example3.png)

![example4](http://www.cs.unc.edu/~eunbyung/abidc/example4.png)

![example5](http://www.cs.unc.edu/~eunbyung/abidc/example5.png)

![example6](http://www.cs.unc.edu/~eunbyung/abidc/example6.png)

![example7](http://www.cs.unc.edu/~eunbyung/abidc/example7.png)

## 4 Challenge
We will release test sets and open the registration soon. Stay tuned!
