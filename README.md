# MonocularDepthEstimation 

This repository contains a Pytorch implementation of monocular depth estimation encoder decoder network based on the network in NeWCRF paper [1]. The encoder is based on Swin Transformer and the decoder is based on the one in [1] taking query and key values from encoder layers. A PPM head is used between the encoder and decoder layers. The network was trained on a subset of the Kitti dataset.

## Features of model in depthmodel.py

1. The encoder is based on a Swin transformer consisting of 4 stages with window size 5 for the first 2 stages, 6 for the 3rd stage and [6, 5] for the 4th stage.
2. The encoder output is passed to a PPM head [3]. 
3. The decoder consists of 4 layers each layer consisting of Neural Window Fully Connected CRF based on the decoder in NeWCRF[1].

## Instructions for training:

1. Change the values of depthdatalist, rawdatalist, valdepthdatalist, valrawdatalist and their respective elements in data_config.py.
2. Run python main.py.

## Results on Kitti dataset 

The model in depthmodel.py was trained on a subset (1846 + augmented images) of Kitti monocular depth estimation dataset. The model was trained for 16 epochs and tested on a validation set of 108 images.
<br/><br/>


| Metric  | Value |
| --- | --- |
| absolute relative error on validation dataset| 0.2034 |


## References

[1] Yuan, W., Gu, X., Dai, Z., Zhu, S., & Tan, P. (2022). Neural window fully-connected crfs for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 3916-3925).

[2] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

[3] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2881-2890).

[4] https://github.com/aliyun/NeWCRFs

[5] https://github.com/berniwal/swin-transformer-pytorch

[6] https://github.com/microsoft/Swin-Transformer

[7] https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678





