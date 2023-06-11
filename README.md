# Industrial Meter Reading 
![](https://github.com/openvinotoolkit/openvino/assets/71766106/72623109-3b68-4c57-ae6d-b1d07839ec29)

This repository hosts the code for training models for the GSoC23 project ["Industrial Meter Reading with OpenVINO"](https://summerofcode.withgoogle.com/programs/2023/projects/3eKcuFkd), which is implemented under Intel's [OpenVINO Toolkit](https://github.com/openvinotoolkit) organization.

## Description
The aim is to build a DeepLearning solution that can read analog meter. This is basically a computer vision project broadly devided into two steps
* Detection : Detecting meter in the frame/image
* Segmentation : Performing semantic segmentation to read scale and pointer.

![](https://github.com/openvinotoolkit/openvino/assets/71766106/be7bd436-d9f7-427e-9cae-043eebdce8c4)

### Model Used
* For detection I used EfficientDet-d0 and trained it with help of TF-model zoo, [here](detector_training.ipynb).
* For segentation I used UNET with EfficientNetb1 backbone, [here](segmentor_training).
  