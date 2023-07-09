# Industrial Meter Reading 
![](https://github.com/openvinotoolkit/openvino/assets/71766106/72623109-3b68-4c57-ae6d-b1d07839ec29)

This repository hosts the code for training models for the GSoC23 project ["Industrial Meter Reading with OpenVINO"](https://summerofcode.withgoogle.com/programs/2023/projects/3eKcuFkd), which is implemented under Intel's [OpenVINO Toolkit](https://github.com/openvinotoolkit) organization.

## Description

The aim is to build a DeepLearning solution that can read analog meter. This is basically a computer vision project broadly devided into two steps
* Detection : Detecting meter in the frame/image
* Segmentation : Performing semantic segmentation to read scale and pointer.

![](https://user-images.githubusercontent.com/71766106/245456398-49028900-a253-46d2-bc3d-7515671f8d6f.gif)

### Segmentor

UNET is one of the most famous semantic segmentation models. It was initially introduced in biomedical image segmentation, such as brain image segmentation and liver image segmentation. But because of its performance, it gets widely used in other image segmentation tasks as well.
According to a research paper [[1](#Refrences)] which shows the performance of widely implemented DL models for semantic segmentation on 'top view person images' mIoU is 80%, 82%, and 84% for FCN, U-Net, and DeepLabv3, respectively.
Since both UNET and DeepLab have almost the same accuracy and inference time, I decided to use UNET for this solution. You can see training [here](training-notebooks/segmentor_training.ipynb).

![](https://github.com/openvinotoolkit/openvino/assets/71766106/45cd4ec8-60ff-49ae-8701-fc5b0548f60c)
![](https://github.com/openvinotoolkit/openvino/assets/71766106/975a5b9e-448c-485e-b23a-420378259ddb)

### Detector

When it comes to object detection, there are a ton of models available, all serving the same purpose hence, it's always hard to choose one. I chose EfficientDet, which is a scalable and efficient object detection model [[2](#Refrences)] and achieves state-of-the-art precision with fewer parameters, as you can see in the below image. I chose its D0 version as I only have to detect one object, and choosing a heavier model would be just overkill. You can see training [here](training-notebooks/detector_training.ipynb).

![](https://user-images.githubusercontent.com/71766106/245152449-2939f7c9-b287-4a4b-9617-f04cb1910bc1.png)

## How to run ?
* Install the dependencies
  * `pip install -r requirements.txt`
* Follow [these](https://github.com/ashish-2005/GSoC23-OpenVINO/wiki/Training-Models#training-on-custom-data) step and [these notebooks](/training_notebooks) if you want to train models on your costum dataset.
* Just run the [OpenVINO inferecne](/OV-meter-reader.ipynb) notebook, if using same models
  * Change path of models with your TF-model in SavedModel [format in notebook](/OV-meter-reader.ipynb).


## Refrences
* [1] Comparison of Deep-Learning-Based Segmentation Models: Using Top View Person Images (2020)
* [2] EfficientDet: Scalable and Efficient Object Detection (2020)
  
