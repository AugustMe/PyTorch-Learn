# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:39:51 2019

@author: ZQQ

普通的卷积神经网络
"""

from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # super方法：CNN继承父类nn.Module的属性，并用父类的方法初始化这些属性
        # nn.Sequential():一个时序容器，可以初始化卷积层、激活层和池化层
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), # 卷积层，(input shape:1*28*28), 
                                    # in_chaneels=1:输入高度; out_channels=16：输出高度,即n_filter; kernel_size=3即filter size；
                                    # kernel_size=5:卷积核大小为5*5（注，也可以为5*4，后面学）,stride=1:滑动窗口步长为1，
                                    # padding=2:输出时不改变输入时候的尺寸，padding=(kernel_size-1)/2 if stride=1
                                    #nn.BatchNorm2d(16),  # 添加批标准化
                                    nn.ReLU() # ReLU激活函数，括号里加inplace=True？？
                                   ) # output shape: 16*28*28

        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2) # 池化层，MaxPool2d最大池化，2*2 采样，窗口滑动步长2
                                   ) # output shape:16*14*14

        self.layer3 = nn.Sequential(nn.Conv2d(16, 32 , 5, 1, 2), # 第二次卷积,参数名对应第一次卷积 ， input shape：16*14*14
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU()
                                   ) # output shape: 32*14*14

        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2) # 第二次池化，input shape: 32*14*14
                                   ) # output shape: 32*7*7

        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, 10), # 全连接层， 输入 32*7*7， 输出10类
                               )
        
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  #flat (batch_size, 32*7*7)
        x = self.fc(x)
        return x

## 查看模型结构
#model = CNN()
#print(model)
    
#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
#            nn.Conv2d(
#                in_channels=1,              # input height 输入高度
#                out_channels=16,            # n_filters  输出高度
#                kernel_size=5,              # filter size
#                stride=1,                   # filter movement/step
#                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
#            ),                              # output shape (16, 28, 28)
#            nn.ReLU(),                      # activation ，激活
#            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
#        )
#        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
#            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
#            nn.ReLU(),                      # activation
#            nn.MaxPool2d(2),                # output shape (32, 7, 7)
#        )
#        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#        output = self.out(x)
#        return output, x    # return x for visualization