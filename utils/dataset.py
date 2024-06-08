#!/usr/bin/python
# coding=utf-8
import os, sys       #os：用于操作文件和目录的模块,sys：提供了对 Python 运行时环境的访问和操作的函数
import torchvision.datasets.folder as datasets  #PyTorch 中用于处理图像数据集的模块，其中 datasets 是该模块的一部分
from PIL import Image                           #用于处理图像的读取、处理和保存

#open打开文件，r只读，b图片，rb打开只读二进制文件
def pil_loader(path, channel=1):   #定义了一个函数 pil_loader，用于加载图像文件
    with open(path, 'rb') as f:       #函数内部使用 open 函数打开文件，
        img = Image.open(f)           #Image.open 函数加载图像文件。然后根据给定的 channel 参数值进行图像转换
        if channel == 1: return img.convert('L')   #如果 channel 的值为 1，则调用 img.convert('L') 将图像转换为灰度图像（单通道）
        if channel == 3: return img.convert('RGB')  #如果 channel 的值为 3，则调用 img.convert('RGB') 将图像转换为 RGB 彩色图像（三通道）
        return img
#读取数据用的，具体用法，看下遍主函数用法
class ClsImageFolder(datasets.ImageFolder):   #定义了一个自定义的数据集类 ClsImageFolder，该类继承自 torchvision.datasets.ImageFolder 类
    #构造函数 __init__ 接受以下参数：
    #root：数据集根目录的路径。
    #transform：对图像进行的数据转换操作。
    #target_transform：对目标标签进行的数据转换操作。
    #channel：图像的通道数，默认为 3。
    #is_valid_file：一个函数，用于确定是否为有效的图像文件，默认为 None。
    def __init__(self, root, transform=None, target_transform=None, channel=3, is_valid_file=None):
        loader = lambda path: pil_loader(path, channel)   #定义了一个 loader 函数，该函数使用之前定义的 pil_loader 函数加载图像文件，并传递给父类的构造函数
        super(ClsImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)  #调用父类 ImageFolder 的构造函数，传递了 root、transform、target_transform、loader 和 is_valid_file 参数

    def _find_classes(self, dir):     #定义了一个 _find_classes 方法，用于从数据集目录中查找类别信息。在该方法中，根据 Python 版本的不同，使用不同的方法获取数据集目录下的子目录名（类别名）
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and not d.name.startswith('.')]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and not d.startswith('.')]
        classes.sort()
        #分别给三个文件上索引，0，1，2
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx       #返回了类别名列表和类别到索引的映射字典
