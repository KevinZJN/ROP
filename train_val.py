#!/usr/bin/python
# coding=utf-8
import time                               #处理时间相关的操作
from utils.dataset import *               #自定义的数据集模块，可能包含数据加载和预处理的功能
import matplotlib.pyplot as plt           #绘制图表和可视化数据
import torch                               #深度学习框架
from tensorboardX import SummaryWriter     #用于创建 TensorBoard 的摘要写入器，以便进行可视化和日志记录。
from torch.utils.data import DataLoader    #用于加载数据的工具类
import torchvision.transforms as transforms #提供的图像变换操作
from tqdm import tqdm                       #在循环中显示进度条
import argparse                             #用于解析命令行参数
import torch.nn as nn                        #PyTorch 中的神经网络模块



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #通过 torch.device 方法确定设备类型，如果当前系统支持 CUDA，则使用 "cuda:0"，否则使用 "cpu"。
print("using {} device.".format(device))    #打印出使用的设备类型，例如 "using cuda:0 device." 表示使用 CUDA 设备进行计算，或者 "using cpu device." 表示使用 CPU 进行计算

#./neu_cls/train
parser = argparse.ArgumentParser(description="Human Parsing")
parser.add_argument('--is_val', type=str,default=False, help='Just val') #表示是否只进行验证，它是一个字符串类型的参数，默认值为 False
parser.add_argument('--train-path', type=str,default=r'240515-ROPdataset\train',
                    help='Path to dataset folder') #指定训练数据集的路径，它是一个字符串类型的参数
parser.add_argument('--val-path', type=str,default=r'240515-ROPdataset\val',
                    help='Path to dataset folder') #指定验证数据集的路径，它是一个字符串类型的参数

parser.add_argument('--batch-size', type=int, default=8, help="Number of images sent to the network in one step.")#每个训练批次中送入网络的图像数量，默认值为 16
parser.add_argument('--workers', type=int, default=0, help="Number of images sent to the network in one step.")#每个训练批次中的线程数
parser.add_argument('--epochs', type =int, default=5, help='Number of training epochs to run')#训练的轮数，默认值为 150
parser.add_argument('--start-lr', type=float, default=0.001, help='Learning rate')#优化器的初始学习率，默认值为 0.001

parser.add_argument('--resume', '-r', action='store_true',default=False, help='resume from checkpoint')  #是否从断点处继续训练，它是一个布尔类型的参数，默认值为 False
parser.add_argument('--checkpoint', type=str, default='')                                               #加载断点训练的权重
#把参数总结起来
args = parser.parse_args()   #解析命令行参数，并将解析结果保存到 args 变量中，以便在后续代码中使用
print(args)

def get_transform():       #函数用于返回一个数据预处理的转换器，其中包含了一系列的图像处理操作，包括随机裁剪、转换为张量和归一化
    tran_size = 224         #设置为 224，表示裁剪后的图像大小
    Normalize = transforms.Normalize(mean=[0.297], std=[0.259])  #Normalize 使用 transforms.Normalize 进行图像的归一化操作，设置均值为 [0.297]，标准差为 [0.259]
    train_transform = transforms.Compose([transforms.RandomResizedCrop(tran_size, scale=(0.9, 1.0)), transforms.ToTensor(), Normalize, ]) #使用 transforms.Compose 组合多个图像处理操作，包括随机裁剪、转换为张量和归一化。最后返回 train_transform
    return train_transform

def get_dataloader():                   #用于返回训练数据集和验证数据集的数据加载器
    data_transform = get_transform()    #调用 get_transform() 函数获取数据预处理的转换器
    train_loader = DataLoader(ClsImageFolder(root=args.train_path, transform=data_transform,channel=3),
                              batch_size=args.batch_size, num_workers=args.workers , #使用 DataLoader 创建训练数据加载器 train_loader，其中使用 ClsImageFolder 类作为数据集，需要提供数据集的根目录、数据预处理的转换器和通道数等参数。同时指定批次大小为 args.batch_size，打乱数据集顺序
                              shuffle=True,
                              )
    test_loader = DataLoader(ClsImageFolder(root=args.val_path, transform=data_transform,channel=3),
                              batch_size=args.batch_size,num_workers=args.workers ,  #使用 DataLoader 创建验证数据加载器 test_loader，参数设置与训练数据加载器类似，但不打乱数据集顺序。
                              shuffle=False,
                              )
    print('train_size after batch: {:4d}  valid_size after batch:{:4d}'.format(len(train_loader), len(test_loader)))  #打印训练数据加载器和验证数据加载器的大小
    return train_loader,test_loader

def val():    #定义了一个名为 val() 的函数，用于进行验证
    global best_acc       #使用 global 关键字声明了一个全局变量 best_acc，用于记录最佳准确率
    net.eval()            #模型设置为评估模式（net.eval()）
    correct = 0.          #，并初始化一些变量，如正确预测的样本数（correct 和 correct5）、总样本数（total）和评估损失（eval_loss）
    correct5 = 0.
    total = 0
    eval_loss = 0.

    with torch.no_grad():#torch.no_grad() 声明一个上下文管理器，该上下文管理器内的计算不会进行梯度反向传播
        for input, targets in tqdm(val_loader):#for 循环中，遍历验证数据加载器 val_loader 中的图像和标签。
            input, targets_ = input.to(device), targets.to(device)#将输入数据和目标标签移动到 CUDA 设备
            outputs = net(input)               #然后将输入数据传递给模型进行前向计算，得到输出
            loss = criterion(outputs, targets_)# 计算损失值

            eval_loss += loss.item()            #损失值累加到 eval_loss 中
            total += targets.size(0)            #通过 targets.size(0) 获取当前批次中的样本数量，并将其累加到 total 中

            _,predicted = torch.max(outputs.data, 1)  #找到输出中的最大值及其对应的索引，与目标标签进行比较
            correct += predicted.eq(targets_.data).cpu().sum()  # 将本批量预测正确的样本数累加预测正确的样本数到 correct 中
            # top_5
            maxk = max((1, 5))               #对于 top-5 准确率的计算，使用 torch.topk() 函数找到输出中最大的五个值及其对应的索引
            yresize = targets_.viefw(-1, 1)
            _,pred = outputs.topk(maxk, 1, True, True)
            correct5 += torch.eq(pred, y_resize).sum().float().item()  #，与目标标签进行比较，累加预测正确的样本数到 correct5 中


        top1_mean_acc = correct / total        #计算了平均准确率 top1_mean_acc，即正确预测的样本数 correct 除以总样本数 total
        top5_mean_acc = correc1t5 / total       #计算了 top-5 准确率 top5_mean_acc，即在前5个预测结果中，正确预测的样本数 correct5 除以总样本数 total
        mean_loss = eval_loss /total           #计算了平均损失 mean_loss，即验证过程中累计的损失值 eval_loss 除以总样本数 total
        total_test_acc.append(top1_mean_acc)   #top1_mean_acc 添加到名为 total_test_acc 的列表中，以便后续统计和分析

        # writer.add_scalar() 将平均损失 mean_loss 和平均准确率 top1_mean_acc 写入到 writer 对象中，以便可视化和记录训练过程中的变化。
        writer.add_scalar('val_meanLOSS', mean_loss, global_step=epoch)
        writer.add_scalar('val_meanACC', top1_mean_acc, global_step=epoch)

        # 如果当前的 top1_mean_acc 大于 best_acc，则更新 best_acc 为当前的准确率，并将模型的状态字典、准确率和当前的迭代次数保存到文件中。
        # 这样可以跟踪和记录达到最佳准确率时的模型状态。
        if top1_mean_acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': top1_mean_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path_best)
            best_acc = top1_mean_acc

    return top1_mean_acc, top5_mean_acc, mean_loss   #函数返回 top1_mean_acc、top5_mean_acc 和 mean_loss

def train():             #定义了一个名为 train() 的函数，用于进行训练
    net.train()          #将模型设置为训练模式（net.train()）
    train_loss = 0       #初始化一些变量，如训练损失（train_loss）、总样本数（total）和正确预测的样本数（correct）
    total = 0
    correct = 0
    for idx, (input, targets) in enumerate(tqdm(train_loader)):      #for 循环中，遍历训练数据加载器 train_loader 中的图像和标签
        input, targets = input.to(device), targets.to(device)              #将输入数据和目标标签移动到 CUDA 设备上
        if hasattr(torch.cuda, 'empty_cache'):                       #用于检查当前 PyTorch 版本是否支持 empty_cache() 函数。如果支持，则调用 torch.cuda.empty_cache() 来清空 CUDA 缓存，释放显存
            torch.cuda.empty_cache()
        outputs = net(input)                                       #并使用模型进行前向计算，得到输出 outputs

        optimizer. zero_grad()  #  # 将模型中参数的梯度设为0，用在优化器上，每一次读取batch_size哥图片去处理
        loss = criterion(outputs, targets)  # 根据输出和类计算交叉熵损失
        loss.backward()  # loss反向传播
        optimizer.step()  # 更新一步梯度下降
        scheduler.step()   #更新学习率

        train_loss += loss.item()                #将每次迭代的损失值 loss.item() 累加到 train_loss
        _, predicted = outputs.max(1)            #使用 outputs.max(1) 找到输出中的最大值及其对应的索引
        total += targets.size(0)                 #通过 targets.size(0) 获取当前批次中的样本数量，并将其累加到 total 中
        correct += predicted.eq(targets).sum().item()    #通过 predicted.eq(targets).sum().item() 统计预测正确的样本数，并累加到 correct 中

    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])          #获取当前的学习率并将其添加到 lr_list 列表中
    Lr = optimizerf.state_dict()['param_groups'][0]['lr']                     #当前学习率赋值给变量 Lr，以在后续的可视化和记录

    train_mealoss = train_loss / len(train_loader)                 #计算训练损失的平均值 train_mealoss，即将 train_loss 除以训练数据加载器 train_loader 的长度，以得到每个样本的平均损失
    train_meanacc = correct / total                                #计算训练准确率 train_meanacc，即将正确预测的样本数 correct 除以总样本数 total，以得到在训练集上的准确率
    writer.add_scalar('Lr', Lr, global_step=epoch)                 #使用 writer.add_scalar() 将学习率 Lr、训练损失 train_mealoss 和训练准确率 train_meanacc 写入到 writer 对象中，以便可视化和记录训练过程中的变化
    writer.add_scalar('train_meanLOSS', train_mealoss, global_step=epoch)    #global_step 参数表示当前的训练步骤或迭代次数，用于指定在可视化中的横坐标位置
    writer.add_scalar('train_meanACC', train_meanacc, global_step=epoch)

    total_train_acc.append(train_meanacc)                 #训练准确率 train_meanacc 添加到名为 total_train_acc 的列表
    return train_meanacc,train_mealoss                    #返回训练准确率和训练损失

# 定义了一个名为 time_sync() 的函数，用于获取准确的时间戳
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():  #通过 torch.cuda.is_available() 判断当前系统是否支持 CUDA。如果支持，则调用 torch.cuda.synchronize() 来同步 CUDA 操作，以确保之前的 CUDA 计算已经完成
        torch.cuda.synchronize()
    return time.time()             #time.time() 函数返回当前时间的时间戳，以秒为单位

if __name__ == '__main__':                 #主程序的逻辑，可以看作是程序的入口

    train_loader,val_loader = get_dataloader()   #调用 get_dataloader() 函数加载训练和验证数据，并得到对应的数据加载器 train_loader 和 val_loader。
    from model.Resnet import ResNet101, ResNet50  # ResNet 模型的不同版本导入
    net = ResNet50().to(device)   #初始化网络模型，这里使用的是 ResNet50 模型，并将其移动到设备 device 上进行计算。

    model='ResNet50'      #对应模型名字进行改动
    starting_epoch=0

    #写入tensorboard
    writer = SummaryWriter(r'result\log/' + model)         #创建一个 SummaryWriter 对象 writer，用于将训练过程中的指标写入到 TensorBoard 日志中，方便可视化和记录训练过程。
    save_path_best =os.path.join(r'result\weight', model+"_best.pth")  #设定模型保存路径 save_path_best，用于保存在验证集上表现最好的模型参数。

    optimizer = torch.optfim.AdamW(net.parameters(), lr=args.start_lr,eps=1e-4)  #AdamW 优化器来优化网络模型的参数，设置初始学习率为 args.start_lr。

    from create_lr_scheduler import create_lr_scheduler           #调用 create_lr_scheduler() 函数创建学习率调度器 scheduler，用于根据训练的进度自动调整学习率
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,   #学习率调度器使用的是自定义的函数 create_lr_scheduler()，根据提供的参数设置学习率调度策略
                                    warmup=True, warmup_epochs=1)
    print("初始化的学习率：", optimizer.defaults['lr'])              #打印初始化的学习率
    lr_list = []

    if args.resume:                   #如果设置了 args.resume 为 True，则加载之前保存的训练检查点，恢复模型训练的状态。加载的检查点路径为 args.checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dictf(checkpoint['net'])
        acc = checkpoint['acc']
        starting_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss().to(device)#交叉熵损失函数 criterion，用于计算模型的损失,将其移动到 CUDA 设备上进行计算

    total_test_acc = []                       #创建了两个空列表 total_test_acc 和 total_train_acc，用于记录训练和测试的准确率
    total_train_acc = []

    tic = time_synqc()                         #记录训练开始的时间

    if args.is_val:                           #args.is_val 的值来确定是进行验证还是训练过程
        top1_mean_acc, top5_mean_acc, mean_loss= val()  #如果 args.is_val 为 True，则执行验证过程。调用 val() 函数来对模型进行验证，并获取验证集上的 top1 准确率 top1_mean_acc、top5 准确率 top5_mean_acc 和平均损失 mean_loss。然后打印这些值
        print('top1_mean_acc:',top1_mean_acc)
        print('top5_mean_acc:', top5_mean_acc)
        print('mean_loss:', mean_loss)
    else:                                     #如果 args.is_val 为 False，则进行训练过程
        best_acc=0                            #首先初始化一些变量，如最佳准确率 best_acc、
        top1_max = 0                          # 最大的 top1 准确率 top1_max、最大的 top5 准确率 top5_max，以及用于记录每个 epoch 的 top1 准确率、top5 准确率和损失的列表
    top5_max = 0
    top1_epoch_mean = []
    top5_epoch_mean = []
    loss_epoch_mean = []

    for epoch in range(starting_epoch, 1 + args.epochs):
            train_mean_acc, train_mean_loss = train()             #遍历从 starting_epoch 到 args.epochs 的每个 epoch。在每个 epoch 中，先调用 train() 函数进行训练，并获取训练集上的平均准确率 train_mean_acc 和平均损失 train_mean_loss
            top1_mean_acc,top5_mean_acc,val_mean_loss = val()     #调用 val() 函数对模型进行验证，并获取验证集上的 top1 准确率 top1_mean_acc、top5 准确率 top5_mean_acc 和平均损失 val_mean_loss。将 top1 准确率转换为浮点型方便后续计算
            top1_mean_acc = float(top1_mean_acc)                #top1_mean_acc是tensor,转换为float方便计算均值

            info1 = 'epoch :{0}/{1}  train_mean_acc :{2}   top1val_mean_acc :{3}  top5val_mean_acc :{4}  ' \
                          'val_mean_loss :{5} ' \
                .format(epoch, args.epochs, train_mean_acc, top1_mean_acc, top5_mean_acc,val_mean_loss)  #首先使用字符串格式化将训练和验证指标信息存储在 info1 变量中，并使用 print 函数将其打印出来
            print(info1)
    top1_epoch_mean.append(top1_mean_acc)     #将 top1 准确率 top1_mean_acc、top5 准确率 top5_mean_acc 和平均损失 val_mean_loss 添加到对应的列表 top1_epoch_mean、top5_epoch_mean 和 loss_epoch_mean 中，以便后续统计和分析
    top5_epoch_mean.append(top5_mean_acc)
    loss_epoch_mean.append(val_mean_loss)

    if top1_max < top1_mean_acc:    #判断当前的 top1 准确率 top1_mean_acc 是否大于 top1_max，如果是，则将 top1_max 更新为 top1_mean_acc
       top1_max = top1_mean_acc
    if top5_max < top5_mean_acc:    #如果当前的 top5 准确率 top5_mean_acc 大于 top5_max，则将 top5_max 更新为 top5_mean_acc
       top5_max = top5_mean_acc

        # 数据可视化
    if args.resume == False:            #通过条件判断 args.resume == False 来确定是否进行数据可视化
            plt.figure(1)                    #使用 plt.figure(1) 创建第一个新的图形窗口
            plt.ploft(range(args.epochs+1), total_train_acc, label='Train Accurancy')  #plt.plot() 函数绘制训练和测试准确率随 epoch 变化的曲线，其中 range(args.epochs+1) 表示 x 轴的取值范围，total_train_acc 和 total_test_acc 分别表示训练和测试准确率的取值，label 参数指定了相应曲线的标签
            plt.plot(range(args.epochs+1), total_test_acc, label='Test Accurancy')
            plt.xlabel('Epoch')                          # plt.xlabel() 和 plt.ylabel() 函数分别设置 x 轴和 y 轴的标签。
            plt.ylabel('Accurancy')
            plt.title('{}-Accurancy'.format(model))      #plt.title() 函数设置图像的标题，其中 {} 会被替换为 model 变量的值
            plt.legend()                                 #plt.legend() 函数添加图例
            plt.savefig('result/{}-Accurancy.jpg'.format(model))  # plt.savefig() 函数将图像保存为文件，文件名中的 {} 会被替换为 model 变量的值。                                 #plt.show() 函数显示图像
            plt.figure(2)                                 #使用 plt.figure(2) 创建第2个新的图形窗口
            plt.plot(range(len(lr_list)), lr_list, color='r', label='lr')   #绘制学习率曲线
            plt.xlabel('Epoch')                           # plt.xlabel() 和 plt.ylabel() 函数分别设置 x 轴和 y 轴的标签。
            plt.ylabel('LR')
            plt.title('{}-LR'.format(model))  # plt.title() 函数设置图像的标题，其中 {} 会被替换为 model 变量的值
            plt.legend()                                  #plt.legend() 函数添加图例
            plt.savefig('result/{}-LR.jpg'.format(model))  # plt.savefig() 函数将图像保存为文件，文件名中的 {} 会被替换为 model 变量的值。
            plt.show()                                    ##plt.show() 函数显示图像

    print(f'val_Best_Acc: {best_acc * 100}%')         #打印最佳准确率
    print(f'top1_max: {top1_max * 100}%')             #最大的 top1 准确率
    print(f'top5_max: {top5_max * 100}%')              #最大的 top5 准确率
    print(f'top1_epoch_mean: {sum(top1_epoch_mean)/len(top1_epoch_mean)}%')  #所有 epoch 上的平均 top1 准确率和平均 top5 准确率
    print(f'top5_epoch_mean: {sum(top5_epoch_mean)/len(top5_epoch_mean)}%')  #所有 epoch 上的平均 top5 准确率
    print(f'loss_epoch_mean: {sum(loss_epoch_mean)/len(loss_epoch_mean)}%')  #所有 epoch 上的平均 损失
    toc = time_sync()                                       #使用 time_sync 函数记录训练结束的时间
    t = (toc - tic) / 3600                                  #计算训练的总时长。最后，使用 print 函数打印训练完成的信息，包括总时长
    print(f'Training Done. ({t:.3f}s)')