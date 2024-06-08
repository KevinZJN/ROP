import torchvision
import torch.nn as nn

# from cl_code.model.shufflenetv2 import shufflenetv2

def alexnet():
    model = torchvision.models.alexnet(pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=9216, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=2048, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=2048, out_features=5, bias=True),
    )
    return model


def vgg16():
    model = torchvision.models.vgg16(pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(),
        # nn.Dropout(p=0.5),
        nn.Linear(4096, 5, bias=True)
    )
    return model

def vgg16_bn():
    model = torchvision.models.vgg16_bn(pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 5, bias=True)
    )
    return model



def googlenet():
    model = torchvision.models.GoogLeNet()  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.fc = nn.Sequential(
        nn.Linear(1024, 8, bias=True)
    )
    return model

def resnet34():
    model = torchvision.models.resnet34(pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.fc = nn.Sequential(
        nn.Linear(512, 8, bias=True)
    )
    return model



def resnet50():
    model = torchvision.models.resnet50(pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.fc = nn.Sequential(
        nn.Linear(2048, 5, bias=True)
    )
    return model

def densenet():
    model = torchvision.models.DenseNet()  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.classifier = nn.Sequential(
        nn.Linear(1024, 8, bias=True)
    )
    return model

def shufflenetv2():
    model = torchvision.models.shufflenet_v2_x1_0( pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.fc = nn.Sequential(
        nn.Linear(1024, 5, bias=True)
    )
    return model


def mobilenetv2():
    model = torchvision.models.mobilenet_v2( pretrained=False)  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(1280, 100, bias=True)
    )
    return model


def efficientnet_b0():
    model = torchvision.models.efficientnet_b0()  # 需要下载预训练模型
    # for param in model.parameters():
    #     param.requires_grrad = False
    # 改写FC
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(1280, 8, bias=True)
    )
    return model



if __name__ == '__main__':
    model = efficientnet()
    print(model)
