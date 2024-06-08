
import torch                  #导入torch库， torch是 PyTorch 深度学习框架的主要模块，提供了各种用于构建和训练神经网络的函数和类。
import torch.nn as nn         #torch.nn 是 PyTorch 中用于定义神经网络模型的模块，提供了各种用于构建神经网络层和模型的类和函数。

import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        # print(1111111111111111)
        return out

def Conv1(in_planes, places, stride=2):    #定义了一个名为 Conv1 的函数，用于创建一个卷积层的序列，并接受3个参数，in_planes：输入通道数，places：输出通道，stride：步幅（默认为 2）
    return nn.Sequential(                  #nn.Sequential 创建一个包含多个层的序列
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False), #定义二维卷积层，输入通道数是 in_planes ，输出通道数是 places，卷积核大小为 7x7，步幅为 stride，填充为 3，且不使用偏置项。
        nn.BatchNorm2d(places),            #二维批归一化层，对输出通道数 places 进行批归一化处理。
        nn.ReLU(inplace=True),             #ReLU 激活函数，将激活值小于零的部分置零，保持非负
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #二维最大池化层，使用 3x3 的池化核进行最大池化操作，步幅为 2，填充为 1。
    )

#定义了一个名为 Bottleneck 的自定义模块，用于创建 ResNet 中的 Bottleneck 残差块。
class Bottleneck_CBAM(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):  #初始化相关参数，in_places：输入通道数，places：输出通道数，stride：步幅（默认为 1），downsampling：是否进行下采样（默认为 False），expansion：扩展倍数（默认为 4）
        super(Bottleneck_CBAM,self).__init__()                # 继承父类的构造函数。
        self.expansion = expansion                       #扩展倍数赋值
        self.downsampling = downsampling                 #下采样数赋值

        self.bottleneck = nn.Sequential(               #定义了一个 self.bottleneck 的序列，其中包含了一系列的卷积层和批归一化层
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False), #一个 1x1 的卷积层，将输入通道数 in_places 转换为输出通道数 places。
            nn.BatchNorm2d(places),                     #二维批归一化层，对输出通道数 places 进行批归一化处理。
            nn.ReLU(inplace=True),                      #ReLU 激活函数，将激活值小于零的部分置零，保持非负
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),  #一个 3x3 的卷积层，将输入通道数 places 转换为相同的输出通道数 places，步幅为 stride，填充为 1
            nn.BatchNorm2d(places),                      #二维批归一化层，对输出通道数 places 进行批归一化处理。
            nn.ReLU(inplace=True),                       #ReLU 激活函数，将激活值小于零的部分置零，保持非负
            CBAMBlock(places),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),  #最后一个 1x1 的卷积层，将输入通道数 places 转换为输出通道数 places * expansion，其中 expansion 是扩展倍数
            nn.BatchNorm2d(places*self.expansion),       #二维批归一化层，对输出通道数 places 进行批归一化处理。
        )

        if self.downsampling:                         #如果 downsampling 参数为 True，则定义一个 self.downsample 的序列
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),  #一个 1x1 的卷积层，将输入通道数 in_places 转换为输出通道数 places * expansion。
                nn.BatchNorm2d(places*self.expansion)        #二维批归一化层，对输出通道数 places 进行批归一化处理。
            )
        self.relu = nn.ReLU(inplace=True)                   #ReLU 激活函数，将激活值小于零的部分置零，保持非负
    def forward(self, x):                                   # #定义了前向传播函数，接受输入张量 x 作为参数。
        residual = x                                        #输入张量 x 保存到 residual 变量中，
        out = self.bottleneck(x)                            #通过 self.bottleneck 序列对输入张量进行一系列的卷积和批归一化操作

        if self.downsampling:                              #如果 downsampling 参数为 True，则对输入张量 x 进行下采样操作，得到 residual
            residual = self.downsample(x)

        out += residual                                    #将 输出张量out 和 张量residual 进行相加
        out = self.relu(out)                               #经过 ReLU 激活函数
        return out                                         #返回输出张量 out。


#本段代码定义了一个 ResNet 模型，它由多个残差层组成，每个残差层都由多个 Bottleneck 块组成。残差块构成的模型能够处理深层网络中的梯度消失问题，并在图像分类任务中取得较好的性能。
class ResNet_CBAM(nn.Module):
    def __init__(self,blocks, num_classes=5, expansion = 4):    #模型的初始化函数，接受blocks、num_classes和expansion作为参数。blocks是一个整数列表，指定每个阶段的残差块数量。num_classes表示模型的输出类别数。expansion是一个倍增因子，用于计算Bottleneck块中通道数的增加。
        super(ResNet_CBAM,self).__init__()                           #调用父类nn.Module的初始化函数，确保正确地初始化模型。
        self.expansion = expansion                              #将输入参数expansion赋值给模型的self.expansion属性。

        self.conv1 = Conv1(in_planes = 3, places= 64)           #创建一个名为conv1的卷积层对象。这里使用了一个自定义的Conv1类，它具有输入通道数为3，输出通道数为64的卷积核。

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1) #创建第一个残差块层layer1。make_layer()方法用于创建包含多个残差块的层。
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2) #创建第二个残差块层layer2。
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)   #创建第三个残差块层layer3
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)  #创建第四个残差块层layer4

        self.avgpool = nn.AvgPool2d(7, stride=1)                                             #创建一个平均池化层，它将输入特征图的大小从7x7减小到1x1。
        self.fc = nn.Linear(2048,num_classes)                                                #创建一个全连接层，用于将最终的特征映射转换为预测的类别分数。

        for m in self.modules():                                                             #对模型的所有模块进行遍历。
            if isinstance(m, nn.Conv2d):                                                     #如果当前模块是nn.Conv2d类型的卷积层，则卷积层的权重进行Kaiming正态分布初始化。
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):                                             #如果当前模块是nn.BatchNorm2d类型的批归一化层
                nn.init.constant_(m.weight, 1)                                              #将当前批归一化层的权重参数初始化为1。
                nn.init.constant_(m.bias, 0)                                                #将当前批归一化层的偏置参数初始化为0

    def make_layer(self, in_places, places, block, stride):                               #定义了一个辅助函数make_layer，用于创建包含多个残差块的层
        layers = []                                                                       #创建一个空列表layers，用于存储残差块
        layers.append(Bottleneck_CBAM(in_places, places,stride, downsampling =True))           #将包含下采样的第一个残差块Bottleneck添加到layers列表中
        for i in range(1, block):                                                         #循环block次，创建剩余的block-1个残差块。
            layers.append(Bottleneck_CBAM(places*self.expansion, places))                       #将普通的残差块Bottleneck添加到layers列表中

        return nn.Sequential(*layers)                                                     #将残差块列表layers转换为nn.Sequential对象，并返回


    def forward(self, x):                             #定义了模型的前向传播函数。
        x = self.conv1(x)                             #将输入x通过卷积层conv1进行卷积操作。

        x = self.layer1(x)                            #将卷积结果x传递给第一个残差块层layer1进行处理。
        x = self.layer2(x)                            #将上一层的输出x传递给第二个残差块层layer2进行处理
        x = self.layer3(x)                            #将上一层的输出x传递给第三个残差块层layer3进行处理
        x = self.layer4(x)                            #将上一层的输出x传递给第四个残差块层layer4进行处理。

        x = self.avgpool(x)                           #将最后一层的输出x通过平均池化层进行池化操作，将特征图大小减小到1x1。
        x = x.view(x.size(0), -1)                     #将池化后的特征图展平为一维向量。
        x = self.fc(x)                                #将展平后的特征向量通过全连接层fc进行线性变换。
        return x                                      #返回最终的输出结果

def ResNet50_CBAM():                                      #返回一个深度为50层的ResNet模型，使用的残差块配置是[3, 4, 6, 3]。
    return ResNet_CBAM([3, 4, 6, 3])

def ResNet101_CBAM():                                     #返回一个深度为101层的ResNet模型，使用的残差块配置是[3, 4, 23, 3]
    return ResNet_CBAM([3, 4, 23, 3])

def ResNet152_CBAM():
    return ResNet_CBAM([3, 8, 36, 3])                    #返回一个深度为152层的ResNet模型，使用的残差块配置是[3, 8, 36, 3]


#一个主程序，用于测试ResNet模型并打印模型的输出形状、计算量和参数量
if __name__=='__main__':                           #条件语句用于判断是否执行主程序
    model = ResNet50_CBAM()                             # 创建一个ResNet50模型的实例
    print(model)                                   #打印ResNet50模型的结构信息

    input = torch.randn(1, 3, 224, 224)           # 创建一个输入张量，大小为1x3x224x224，用于模型的前向传播。
    out = model(input)                            # 将输入张量传递给ResNet50模型进行前向传播，得到输出张量out。
    print(out.shape)                              #打印输出张量的形状
    from thop import profile                      #导入thop库中的profile函数，用于计算模型的计算量和参数量。
    flops, params = profile(model, inputs=(input,)) #使用profile函数计算ResNet50模型的计算量和参数量。
    print('flops', flops)                          ## 打印计算量
    print('params', params)                        ## 打印参数量
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0)) #打印模型的计算量和参数量，以G和M为单位进行格式化输出。