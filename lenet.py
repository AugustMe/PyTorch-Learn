# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:48:12 2019

@author: ZQQ

这里针对数据集为MNIST，为MNIST搭建的网络

MNIST数据：图片是黑白的，每张图片大小为1*28*28（1：灰度图片，RGB图片则为3）

我们设置padding=2，原始图片填充为28+2*2=32，所以对应输入层32*32，这样卷积后的特征图还是28*28
"""

from torch import nn

class LeNet(nn.Module):
    def __init__(self): # 官方定义的写法，括号中还可以写输入的维度，输出的维度等
        super(LeNet,self).__init__() # super方法：LeNet继承nn.Module的属性，并用父类的方法初始化这些属性
        # 在算层数的时候，一般把输入层去掉,lenet-5:7层
        
        ### 第一层：卷积层
        # input shape:1*28*28
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, # 输入通道（输入数据体[图像]的深度）
                                             out_channels=6, # 输出通道（输出数据体[图像]的深度，有6个卷积核，生成6个feature maps
                                             kernel_size=5, # 滤波器（卷积核）的大小，即kernel_size=(5,5)
                                             stride=1, # 滑动的步长
                                             padding=2, # 四周进行1个像素点的0填充，如果想输出尺寸=输出尺寸，设置padding= (kernel_size-1)/2, if stride=1
                                             dilation=1, # 卷积对于输入数据体的空间间隔，默认为1
                                             groups=1, # 输出数据体和输入数据体上的联系，默认为1，也就是所有的输出和输入都是相关联的，如果为2，则表示输入的深度被分割成2部分，输出的深度也被分割成2部分，它们之间分别对应起来
                                             bias=True)) # 布尔值，默认为True，表示使用偏置
        # output shape:6*28*28
        
        ### 第二层：池化层，选择最大值池化,对输入的特征图进行压缩，一方面使特征图变小，简化网络计算复杂度；另一方面进行特征压缩，提取主要特征
        # input shape：6*28*28
        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, # 同上
                                                 stride=2, # 同上
                                                 padding=0, # 不进行0填充
                                                 dilation=1, # 同上
                                                 return_indices=False, # 返回最大值所处的下标，默认为False
                                                 ceil_mode=False)) # 表示使用一些方格代替层结构，默认为False，一般不设置
        # output shape: 6*14*14
        
        ### 第三层：卷积层
        # input shape: 6*14*14
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=6,
                                              out_channels=16,
                                              kernel_size=5, 
                                              stride=1,
                                              padding=0)) # 默认为0，不进行填充
        # output shape: 16*10*10
        
        ### 第四层：池化层
        # input shape:16*10*10
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,
                                                 stride=2))
        # output shape:16*5*5
        
        ### 第五层：全连接层fc1
        self.layer5 = nn.Sequential(nn.Linear(16*5*5,120)) # 没有ReLU
        
        ### 第六层：全连接层fc2
        self.layer6 = nn.Sequential(nn.Linear(120,84))
        
        ### 第器层：全连接层fc3
        self.layer7 = nn.Sequential(nn.Linear(84,10)) # 输出10类
        
        # 网络前向传播过程
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1) #全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        x = self.layer5(x)          
        x = self.layer6(x)
        x = self.layer7(x)
        return x
        
#model = LeNet()
#print(model)   
        
        
        
        