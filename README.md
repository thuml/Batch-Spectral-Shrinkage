# Batch-Spectral-Shrinkage
Code release for Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning (NeurIPS 2019)

# Batch-Spectral-Shrinkage

## Prerequisites:

* Python3
* PyTorch == 0.4.0/0.4.1 (with suitable CUDA and CuDNN version)
* torchvision >= 0.2.1
* Numpy
* argparse
* PIL

## Dataset:

Stanford Dogs can be found here: http://vision.stanford.edu/aditya86/ImageNetDogs/

FGVC Aircraft can be found here: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Stanford cars can be found here: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

CUB-200-2011 can be found here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

Oxford-IIIT Pet can be found here: https://www.robots.ox.ac.uk/~vgg/data/pets/


## Training on one dataset:

run this command to run this code: 

python BSS.py --gpu_id 0 --trainpath 'trainpath' --testpath 'testpath' --method l2 --lr 0.01

change 'trainpath' to path of your training dataset, 'testpath' to path of your testing dataset, method is 'l2' or 'l2+bss',lr is 0.01 or 0.001.


## Citation:

If you use this code for your research, please consider citing:

```
@incollection{NIPS2019_8466,
title = {Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning},
author = {Chen, Xinyang and Wang, Sinan and Fu, Bo and Long, Mingsheng and Wang, Jianmin},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {1906--1916},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8466-catastrophic-forgetting-meets-negative-transfer-batch-spectral-shrinkage-for-safe-transfer-learning.pdf}
}

```
## Contact
If you have any problem about our code, feel free to contact chenxinyang95@gmail.com.
