# Image Classification via Dynamic Convolution 

In this repo,I implemented the following paper in **Pytorch**: </br>
[Dynamic Convolution: Attention over Convolution Kernels.](https://arxiv.org/abs/1912.03458)</br>

I implemented dynamic convolution for Resnet.</br>
</br>
#### **Executing code**: </br>
-  Go the current directory</br>
-  Execute the following command : (these arguments are default arguments)</br>
>python3 main.py --dataset 'cifar10' --batch-size 128 --test-batch-size 20 --epochs 160 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --net-name dy_resnet18 </br>
</br>
#### **Files** : </br>
 
- main.py - Source file </br>
-  dy_conv.py - Consists of 2d- Dynamic convolution class </br>
-  dy_resnet.py - Implemented Resnet with Dynamic Convolution.</br>
- load.py - Code to load the saved models.</br>
-  Dynamic Convolution.pdf - Report</br>