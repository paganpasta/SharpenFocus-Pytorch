# Sharpen Focus: Learning With Attention Separability and Consistency
## A Pytorch implementation

```diff
- Unofficial repository
```
Doesn't appear to provide any significant improvment over baseline. 
The implementation may have errors, please be careful.

I'll be working on adding some visualisations to verify the attention maps.
## Results

| Dataset | CIFAR-10  | CIFAR-100  |  STL-10 |
|:-------:|:---------:|:----------:|:-------:|
|Resnet-18|   94.25   |   73.32    |  81.85  |
|SFocus-18|   94.16   |   71.30    |  82.77  |


### Requirements

`Pytorch 1.2`

### Usage

`python main.py --dataset cifar10 --batch-size 128 --prefix run0 --epochs 300 --milestones 75 150 225`


### Acknowledgement
This work is built upon the following repositories:

1. [CBAM](https://github.com/Jongchan/attention-module)
2. [GAIN](https://github.com/ngxbac/GAIN)

### Bibtex

Paper
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Lezi and Wu, Ziyan and Karanam, Srikrishna and Peng, Kuan-Chuan and Singh, Rajat Vikram and Liu, Bo and Metaxas, Dimitris N.},
title = {Sharpen Focus: Learning With Attention Separability and Consistency},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
} 
```

Code
```
@misc{sfocus,
  author = {Singh, Aditya},
  title = {Sharpen Focus: Learning With Attention Separability and Consistency},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MacroMayhem/SharpenFocus-Pytorch}}
}
```
