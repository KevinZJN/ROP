import torchvision
import torchvision.transforms as transforms        #PyTorch 中用于图像转换的模块
from torchvision import datasets                    #PyTorch 中包含一些常用数据集的模块
from torch.utils.data import DataLoader              #PyTorch 中用于加载数据的工具类


class FloderData():
    def __init__(self, batch_size, train_path, valid_path, test_path, num_w, load_model):
        self.bs = batch_size                        #批量大小
        self.train_path = train_path                #训练数据集的路径
        self.vaild_path = valid_path                #验证数据集的路径
        self.test_path = test_path                  #测试数据集的路径
        self.num_w = num_w                          #用于加载数据的工作进程数
        self.load_model = load_model                #加载模式，可以是 'train' 或 'test'
        #定义了一个名为 image_transforms 的字典，其中包含了不同数据集加载模式下的图像转换操作
        self.image_transforms = {
            # Train uses data augmentation
            'train':               #train' 模式下的图像转换操作包括随机裁剪、随机旋转、颜色调整、随机水平翻转、中心裁剪、转换为张量、归一化等操作
                transforms.Compose([
                    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=224),  # Image net standards
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])  # Imagenet standards
                ]),
            # Validation does not use augmentation
            'valid':                #'valid' 模式下的图像转换操作包括缩放、中心裁剪、转换为张量、归一化等操作。
                transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'test':                #'test' 模式下的图像转换操作包括缩放、转换为张量、归一化等操作
                transforms.Compose([
                    transforms.Resize(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }

    def get_dataloader(self):
        if self.load_model == 'train':
            # Datasets from folders
            #ImageFolder函数root：在root指定的路径下寻找图片，transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
            data = {
                'train':
                    datasets.ImageFolder(root=self.train_path, transform=self.image_transforms['train']),
                'valid':
                    datasets.ImageFolder(root=self.vaild_path, transform=self.image_transforms['valid']),
            }
             #如果 self.load_model 的值是 'train'，则创建训练数据集和验证数据集的数据加载器。首先，使用 datasets.ImageFolder 类分别加载训练数据集和验证数据集，
            # 传递了对应的根目录和图像转换操作。然后，使用 DataLoader 类创建数据加载器，将训练数据集和验证数据集的数据集对象作为参数，并设置批量大小、工作进程数和是否洗牌。
            # 最后，将数据加载器存储在 dataloaders 字典中
            # Dataloader iterators, make sure to shuffle
            dataloaders = {
                'train': DataLoader(data['train'], batch_size=self.bs, num_workers=self.num_w, shuffle=True),
                'valid': DataLoader(data['valid'], batch_size=1, num_workers=self.num_w, shuffle=False)
            }
            #如果 self.load_model 的值是 'test'，则创建测试数据集的数据加载器。类似地，使用 datasets.ImageFolder 类加载测试数据集，
            # 并使用 DataLoader 类创建数据加载器。同样，将数据加载器存储在 dataloaders 字典中
        elif self.load_model == 'test':
            # Datasets from folders
            data = {
                'test':
                    datasets.ImageFolder(root=self.test_path, transform=self.image_transforms['test']),
            }

            # Dataloader iterators, make sure to shuffle
            dataloaders = {
                'test': DataLoader(data['test'], batch_size=self.bs, num_workers=self.num_w, shuffle=False),
            }
        else:#如果 self.load_model 的值既不是 'train' 也不是 'test'，则打印错误消息
            print('model type choose error')
        return dataloaders        #返回 dataloaders 字典，其中包含了相应的数据加载器
