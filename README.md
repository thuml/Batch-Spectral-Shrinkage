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

run this command to run this code: ($L^2$+BSS)

python BSS.py --trainpath 'trainpath' --testpath 'testpath' --classnum 120

change 'trainpath' to path of your training dataset, 'testpath' to path of your testing dataset, classnum to your num of class in the dataset you choose.

## Training on one dataset:

All the parameters are set as the same as parameters mentioned in the article. 
You can use the following commands to the tasks:

python -u train.py --gpu_id n --src src --tgt tgt

n is the gpu id you use, src and tgt can be chosen as in "dataset_list.txt".

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
