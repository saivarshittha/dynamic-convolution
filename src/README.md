# Image Classification via Dynamic Convolution 

In this repo,I implemented the following paper in **Pytorch**: </br>
[Dynamic Convolution: Attention over Convolution Kernels.](https://arxiv.org/abs/1912.03458)

I implemented dynamic convolution for Resnet.</br>
Executing code : </br>
1. Go the current directory</br>
2.Execute the following command : (these arguments are default arguments)</br>
python3 main.py --dataset 'cifar10' --batch-size 128 --test-batch-size 20 --epochs 160 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --net-name dy_resnet18 </br>
Files :</br>
1. main.py - Source file </br>
2. dy_conv.py - Consists of 2d- Dynamic convolution class </br>
3. dy_resnet.py - Implemented Resnet with Dynamic Convolution.</br>
4. load.py - Code to load the saved models.</br>
5. Dynamic Convolution.pdf - Report</br>