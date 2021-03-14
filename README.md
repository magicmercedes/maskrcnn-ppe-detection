# Personal Protective Equipment (PPE) detection using Mask R-CNN


![prediction_2](https://user-images.githubusercontent.com/29149625/111077421-f9a09400-84f0-11eb-999d-73f792496f31.png "ppe-prediction")

## Table of Contents 
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Making Dataset](#making-dataset)
4. [Train the model](#train-the-model)
5. [Inspecting the model](#inspecting-the-model)

## Introduction
The codes are based on implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) by ([https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)) on Python 3, Keras, and TensorFlow to automatically detect Personal Protective Equipment (PPE) such as helmet, noise cancelling headphones, vest, eye safety goggles etc.

- It runs in Google colab [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges) (GPU enabled environment) using Matterport's Mask_RCNN framework.

#**Dataset**
The model is trained using the PPE dataset (http://aimh.isti.cnr.it/vw-ppe/ , https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7CBGOS)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          . I picket a total 60 PPE images (40 train and 20 validation). We are not demanding high accuracy from this model, so the tiny dataset should suffice. A ppe dataset of 50 images is created which include 4 classes;
 -  Helmet
 -  Vest
 -  Googles
 - Earmuff


    

The repository includes:

-   Source code forked from the official Mask R-CNN repository.
-   The self annonated ppe Dataset
-   Instructions on how to train from scratch your network
-   A google collab notebook to visualize the detection pipeline
- Pre-trained weights on MS COCO

*For more information on [Mask R-CNN](https://arxiv.org/abs/1703.06870) please visit their official github repo ([https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)) or their paper.*



*All code has been tested under Tensorflow 1.xx and keras 2.15*

## Installation
 1. Clone the repo `git clone https://github.com/matterport/Mask_RCNN `
 2. install dependencies
	  `cd Mask_RCNN`
	  `pip3 install -r requirements.txt` (
	  *(Note: the versions I used here keras == 2.1.5 and Tensorflow == 1.5)*
 3. Run Setup `python3 setup.py install`

**_implementing the Mask RCNN architecture without getting any errors._**



    !pip install keras==2.2.5  
    %tensorflow_version 1.x

## Making Dataset

1. Find all images into one single folder which are to be used for training
2. Create annotation in makesense.ai *(save it in .json format)*
3. Create "val" and "train" folder having structure defined below.

Dataset folder structure:
```
dataset
|- train
  |- pic1.jpg
  |- pic2.jpg
  |- ...
  |- via_region_data.json
|- val
  |- pic3.jpg
  |- pic4.jpg
  |- ...
  |- via_region_data.json
```
*Make sure you include corresponding annotations(.json) in correct directory.*

![rsz_test_label](https://user-images.githubusercontent.com/29149625/111077849-f8706680-84f2-11eb-8502-b5cfbccfb923.jpg)

## Train the model


To start training the network you can create your own config file or modify ppe.py.
Instead of training a model from scratch, we use the COCO pre-trained model_ (mask_rcnn_coco.h5) _as checkpoint to perform transfer lerarning_

    #Train a new model starting from pre-trained COCO weights
    python ppe.py train — dataset=C:/../dataset — weights=coco
    
## Inspecting the Model 
- The weights will be saved every epoch in the logs folder
- [ppe_inference.ipynb](https://github.com/Allopart/Maritme_Mask_RCNN/blob/master/mrcnn/maritime.ipynb) to visualize the results through Jupyter Notebook. Some results are shown as follows:

![prediction_4](https://user-images.githubusercontent.com/29149625/111077451-205eca80-84f1-11eb-9d0d-4c824aba6483.png)

![prediction_1](https://user-images.githubusercontent.com/29149625/111077466-2eace680-84f1-11eb-8239-88272785d847.png)

![prediction_5](https://user-images.githubusercontent.com/29149625/111077486-41bfb680-84f1-11eb-88bc-651c6e16b4d3.png)
