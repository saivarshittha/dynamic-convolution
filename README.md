# Image Classification via Dynamic Convolution 

In this repo,I implemented the following paper in **Pytorch**: </br>
[Dynamic Convolution: Attention over Convolution Kernels.(CVPR2020)](https://arxiv.org/abs/1912.03458)</br>

I implemented dynamic convolution for Resnet.</br>
Find the report and discussion here : [Report](https://docs.google.com/document/d/1IJKyyz8CP__eX7hGutsVHpdH8BZd33EKCoHaILujwRQ/edit?usp=sharing)</br>
</br>
**Execute Code** : </br>
-  Go the current directory</br>
-  Execute the following command : (these arguments are default arguments)</br>
>python3 main.py --dataset 'cifar10' --batch-size 128 --test-batch-size 20 --epochs 160 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --net-name dy_resnet18 </br>
</br>

 **Files** :
- /src/main.py - Source file </br>
- /src/dy_conv.py - Consists of 2d- Dynamic convolution class </br>
- /src/dy_resnet.py - Implemented Resnet with Dynamic Convolution.</br>
- /src/load.py - Code to load the saved models.</br>
-  Dynamic Convolution.pdf - Report</br>
- /observations/cifar10-dy_resnet18.txt - output while training cifar10 on dy_resnet18 50 epochs : Best accuracy - 82.46
- /observations/cifar10-dy_resnet18-50 epochs.txt - output while training cifar10 on dy_resnet18 50 epochs : Best accuracy - 82.86
- /observations/cifar100-dy_resnet18.txt -  output while training cifar100 on dy_resnet18 40 epochs : Best accuracy - 52.85
- /src/cub-200-2011.py - dataloader for [cub-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset. (Apparently, I did by mounting on drive)
