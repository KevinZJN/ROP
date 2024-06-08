
# -*- coding: utf-8 -*-
import torch                         #进行深度学习模型的训练和推理
import os                             #用于与操作系统进行交互，如文件路径的操作
from torchvision.datasets import ImageFolder         #来自torchvision.datasets模块的类，用于加载图像文件夹数据集
from sklearn.metri1cs import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score      #来自sklearn.metrics模块的函数，用于计算分类模型的性能指标，如ROC曲线、AUC、分类报告和准确率
from torchvision import transforms                   #来自torchvision模块的类，用于定义图像的预处理操作
import numpy as np                                   #用于进行数值计算和数组操作
import matplotlib.pyplot as plt,                      #用于绘制图形和可视化数据的Matplotlib子模块


#作用是计算和显示混淆矩阵的图像，并输出准确率和分类报告，用于评估分类模型的性能。
def confusion_matrix_F1(y_true,y_pred,num_class):                                #输入参数为 y_true 和 y_pred，分别表示真实标签和预测标签
    cm = confusion_matrix(y_true, y_pred,labels=None,sample_weight=None)  #confusion_matrix 函数计算混淆矩阵 cm，其中 labels 和 sample_weight 的值为 None。
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]               #将混淆矩阵的值归一化到 [0, 1] 范围，即每一行的和为1，通过除以每行的总和实现
    print(cm)
     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)            #使用 plt.imshow 绘制归一化后的混淆矩阵图像，
    plt.title('Confusion Matrix')                                         #设置标题
    plt.colo1rbar()                                                        #设置颜色条和坐标轴标签
    tick_marks = np.arange(num_class)                                     #使用 np.arange 创建横坐标刻度和纵坐标刻度
    plt.xticks(tick_marks, classes,rotation = 90,fontsize=10)             #使用 plt.xticks 和 plt.yticks 设置横坐标和纵坐标的刻度标签，并进行一些显示设置。
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')                                         #使用 plt.xlabel 和 plt.ylabel 设置坐标轴的标签
    plt.ylabel('True Label')
    plt.show()                                                            #使用 plt.show 显示绘制的混淆矩阵图像

    fig, ax = plt.subplots()                                              #接下来，使用 plt.subplots 创建一个图形和一个坐标轴对象 ax，
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)        #使用 ax.imshow 绘制归一化后的混淆矩阵图像
    ax .figure.colorbar(im, ax=ax)                                         #使用 ax.figure.colorbar 添加颜色条
    ax.set(xticks=np.arange(cm.shape[1]),                                 #使用 ax.set 设置坐标轴的刻度、标签和其他属性
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True',
           xlabel='Predicted')

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",               #使用 plt.setp 对坐标轴刻度标签进行设置，包括旋转、对齐和字体大小等
             rotation_mode="anchor")

    normalize = True
    fmt = '.2f' if normalize else 'd333'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):                                         #使用一个嵌套的循环遍历混淆矩阵的每个元素，在图像上添加文本显示混淆矩阵的数值
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()                                                   #使用 fig.tight_layout 调整图形布局
    plt.  show()                                                           #使用 plt.show 显示绘制的混淆矩阵图像

    print('Accuracy:{:.3f}'.format(accurac1y_score(y_true, y_pred)))      #使用 accuracy_score 函数计算准确率，并使用 print 打印准确率
    pre = classification_report(y_true, y_pred, target_names=classes, digits=5)   #使用 classification_report 函数计算分类报告，并传入参数 target_names 和 digits，然后使用 print 打印分类报告
    print(pre)

    #开始计算精确度
    # pre = []
    # for i in range(num_class):
    #     tp = cm[i][i]
    #     fp = np.sum(cm[:, i]) - tp
    #     pre1 = tp / (tp + fp)
    #     pre.append(pre1)
    # print('单类别的精确度', pre)
    # print('平均精确度', np.mean(pre))

    #开始计算灵敏度==召回率
    sen = []
    for i in range(num_class):
        tp = cm[i][i]
        fn = np.sum(cm[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    print('单类别的灵敏度', sen)
    print('平均灵敏度', np.mean(sen))

    #开始计算特异性
    spe = []
    for i in range(num_class):
        number = np.sum(cm[:, :])
        tp = cm[i][i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = number + tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    print('单类别的特异性', spe)
    print('平均特异性', np.mean(spe))

    #开始计算准确度
    acc = []
    for i in range(num_class):
        number = np.sum(cm[:, :])
        tp = cm[i][i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
    print('单类别的准确度', acc)
    print('平均准确度', np.mean(acc))


def get_transform_for_test():                           #定义一个在测试阶段使用的数据转换操作，用于对测试数据进行预处理，包括剪裁、转换为Tensor格式和归一化等操作
    Normalize = transforms.Normalize(mean=[0.297], std=[0.259])           #定义了一个 Normalize 变量，使用 transforms.Normalize 函数来进行数据归一化操作。mean=[0.297] 和 std=[0.259] 分别表示要减去的均值和除以的标准差
    #transforms.RandomResizedCrop(224, scale=(0.9, 1.0))：随机剪裁图像为指定的大小，224表示目标图像的大小，scale=(0.9, 1.0)表示随机剪裁的范围。
    #transforms.ToTensor()：将图像转换为Tensor格式。
    #Normalize：对图像进行归一化操作
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), transforms.ToTensor(), Normalize])  #transforms.Compose 函数创建一个数据转换的组合操作
    return train_transform              #返回定义好的数据转换操作 train_transform

#代码的作用是使用训练好的模型对测试集进行预测，并计算混淆矩阵、准确率、分类报告以及绘制 ROC 曲线并计算 AUC。最终，将绘制的 ROC 曲线保存为图像文件并显示在图形界面中
def test(model,num_class):                         #输入参数包括 model 和 weight_path，分别表示待测试的模型和模型权重的路径
    test_dir = os.path.join(data_root, 'val')        #表示测试集的路径
    class_list = list(os.listdir(test_dir))           #使用 os.listdir 获取测试集目录下的类别列表
    class_list.sort()                                 #，并对其进行排序
    transform_test = get_transform_for_test()         #调用 get_transform_for_test 函数获取测试数据的转换操作

    test_dataset = ImageFolder(test_dir, transform=transform_test)    #使用 ImageFolder 创建 test_dataset，并指定数据转换操作为 transform_test
    test_loader = torch.utils.data.DataLoader(                         #使用 torch.utils.data.DataLoader 创建测试数据加载器 test_loader，设置了批量大小为 1，不进行洗牌，不丢弃最后一个不完整的批次，使用 GPU 加速，设置工作进程数为 1
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)

    # checkpoint = torch.load(weight_path,map_location='cuda:0')         #使用 torch.load 加载模型权重，将其存储在 checkpoint 变量中
    # model.load_state_dict(checkpoint['net'])                           #使用 model.load_state_dict 加载模型权重到待测试的模型中
    model.load_state_dict(torch.load('pth/ResNet_revise.pth',map_location='cuda:0'))
    model.eval()                                                       #模型设置为评估模式，即 model.eval()

    score_list = []                                                    #函数初始化一些空列表 score_list、label_list、y_pred 和 y_true，用于存储预测得分、标签和预测结果
    label_list = []
    y_pred=[]
    y_true=[]
    for i, (inputs, labels) in enumerate(test_loader):                #使用 enumerate 遍历测试数据加载器 test_loader 的每个批次
        inputs = inputs.to(device)                                        #在每个批次中，将输入数据和标签移至 GPU 上（如果可用）
        labels = labels.to(de1vice)

        outputs = model(inputs)                                       #通过模型进行前向传播得到输出
        prob_tmp = torch.nn.Softmax(dim=1)(outputs)                   # 使用 torch.nn.Softmax 对输出进行 softmax 操作得到类别概率

        _, predicted = torch.max(outputs.data, 1)                     #使用 torch.max 函数找到概率最大的类别作为预测结果，
        y_pred.extend(predicted.cpu().numpy())                        #预测结果和真实标签转移到 CPU 上并转换为 NumPy 数组格式
        y_true.extend(labels.cpu().numpy())                           #将预测结果和真实标签分别添加到 y_pred 和 y_true 列表中

        score_tmp = prob_tmp  # (batchsize, nclass)
        score_list.extend(score_tmp.detach().cpu().numpy())          #类别概率转移到 CPU 上并转换为 NumPy 数组格式，并添加到 score_list 列表中
        label_list.extend(labels.cpu().numpy())
    #遍历完所有批次后，y_pred 和 y_true 列表分别存储了所有预测结果和真实标签
    print(y_pred)       #打印预测标签和真实标签
    print(y_true)
    confusion_matrix_F1(y_true,y_pred,num_class)                              #函数调用 confusion_matrix_F1 函数，传递了 y_true 和 y_pred 作为参数，计算并打印混淆矩阵

    score_array = np.array(score_list)                              #函数将 score_list 转换为 NumPy 数组 score_array
    label_tensor = torch.tensor(label_1list)                         #将 label_list 转换为张量 label_tensor，然后重塑张量的形状为 (样本数, 1)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)    #使用 torch.zeros 创建具有正确形状的全零张量 label_onehot，并使用 scatter_ 方法将标签张量的值分散到 label_onehot 张量中。
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)                            #最后，将 label_onehot 转换为 NumPy 数组

    print("score_array:", score_array.shape)                         # 打印得分形状
    print("label_onehot:", label_onehot.shape)

    fpr_dict = dict()                                                #创建了三个空字典 fpr_dict、tpr_dict 和 roc_auc_dict，用于存储每个类别的 FPR、TPR 和 AUC
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):                                       #遍历 range(num_class)，计算每个类别的 FPR 和 TPR，并使用 sklearn.metrics.roc_curve 函数计算每个类别的 ROC 曲线
        fpr_dict[i], tpr_dict[i], _ = roc_curve(
            label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])              #roc_curve 函数的输入参数是真实标签的独热编码形式 label_onehot[:, i] 和预测得分的第 i 列 score_array[:, i]，返回该类别的 FPR、TPR 和阈值
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(             #使用 sklearn.metrics.auc 函数计算每个类别的 AUC 值，将其存储在 roc_auc_dict 字典中
        label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    all_fpr = np.unique(n1p.concatenate(                             # np.concatenate 函数将每个类别的 FPR（fpr_dict[i]）合并为一个数组
        [fpr_dict[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)                               #使用 np.unique 函数获取合并后数组中的唯一值，并将其赋值给变量 all_fpr
    for i in range(num_class):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])    #然后，通过遍历每个类别的索引 i，使用 np.interp 函数在 all_fpr 上进行插值计算。np.interp 函数的输入参数为 all_fpr（要插值的横坐标数组）、fpr_dict[i]（已知的横坐标数组）和 tpr_dict[i]（已知的纵坐标数组）。插值计算得到的纵坐标结果累加到 mean_tpr 中
    mean_tpr /= num_class                                           # mean_tpr 除以类别数 num_class，得到宏平均的平均 TPR
    fpr_dict["macro"] = all_fpr                                     #将 all_fpr 存储在 fpr_dict["macro"] 中，将 mean_tpr 存储在 tpr_dict["macro"] 中。
    tpr_dict["macro"] = mean_tpr                                    #这样，fpr_dict["macro"] 和 tpr_dict["macro"] 分别存储了宏平均的 FPR 和 TPR

    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])  #使用 auc 函数计算宏平均的 AUC 值，将结果存储在 roc_auc_dict["macro"] 中
    print("AUC",   roc_auc_dict["macro"])
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],                     #使用 plt.plot 函数绘制微平均的 ROC 曲线，其中横坐标是微平均的 FPR（fpr_dict["micro"]），纵坐标是微平均的 TPR（tpr_dict["micro"]）
             label='ROC curve (AUC = %0.2f)' % roc_auc_dict["macro"],  #标签显示了曲线的 AUC 值（roc_auc_dict["macro"]），线条颜色为深粉色（'deeppink'），线型为实线（'-'），线宽为2
             color='deeppink', linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')       #使用 plt.plot 函数绘制一条对角线，代表随机猜测的ROC曲线。该对角线从 (0,0) 到 (1,1)，颜色为海军蓝（'navy'），线宽为2，线型为虚线（'--'）

    x3 = np.linspace(0.118, 0.1188, 1000)                              # 生成一个包含一系列 x 值的数组 x3，以及两个包含一系列 y 值的数组 y1_new 和 y2_new
    y1_new = np.linspace(0.90, 0.905, 1000)                            #使用 np.isclose 函数判断 y1_new 和 y2_new 是否在给定的公差范围内相似，并使用 np.argwhere 函数找到相似的索引
    y2_new = np.linspace(0.90, 0.905, 1000)
    idx = np.argwhere(np.isclose(y1_new, y2_new, atol=0.1)).reshape(-1)
    if x3.all() == 1 - y2_new.all():                                   #如果 x3 全部等于 1 减去 y2_new 的对应值，那么输出 x3 中对应的索引值，表示等误识率（Equal Error Rate）
        print("eer:", x3[idx])

    plt.xlim([-0.02, 1.0])                                             #设置 x 轴和 y 轴的坐标范围，并设置 x 轴和 y 轴的标签。
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", bbox_to_anchor=(0.98, 0.05), fontsize=13)  #使用 plt.legend 函数添加图例，并设置图例的位置和字体大小
    plt.savefig('result/roc.png', dpi=600)                                          #最后，使用 plt.savefig 函数保存 ROC 曲线图像为 'roc.png'，设置图像的 DPI 为 600，并使用 plt.show 函数显示图像
    plt.show()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"                          #设置环境变量，指定可见的 CUDA 设备为设备号为 0 的 GPU。这可以用来控制程序在具有多个 GPU 的系统上使用哪个 GPU 运行
classes = ['0','1','2','3'
           ]     #classes 是一个包含类别名称的列表，其中包含了 'daisy'、'dandelion'、'roses'、'sunflowers' 和 'tulips'
num_class = 17                                                     #num_class 是类别的数量，这里设置为 5
data_root = r'dataset'                #data_root 是数据集的根目录，指定了花卉数据集的根目录路径
# test_weights_path = r"result\weight\ResNet50_best.pth"  #test_weights_path 是测试时使用的模型权重的路径，指定了 ConvMixer 模型的最佳权重文件的路径。
gpu = "cuda:0"                                                     #gpu 是指定要在哪个 GPU 上运行代码的标识符，这里设置为 "cuda:0"，表示在设备号为 0 的 GPU 上运行代码。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #通过 torch.device 方法确定设备类型，如果当前系统支持 CUDA，则使用 "cuda:0"，否则使用 "cpu"。
if __name__ == '__main__':
    from model import *
    model = ResNet_re1vise(num_classes=17).to(device)
    # model.load_state_dict(torch.load('pth/VGG.pth'))
    # net = ResNet50().to(device)   #初始化网络模型，这里使用的是 ResNet50 模型，并将其移动到设备 device 上进行计算。
    # device = torch.device(gpu)                                   #使用 torch.device 函数将 gpu 变量（"cuda:0"）转换为 device 对象，用于指定代码在哪个设备上运行
    # model = net.to(device)                                       #将 net 模型移动到指定的设备上，使用 model = net.to(device)
    test(model,num_class)                               #调用 test 函数，传入移动到指定设备的模型 model 和测试时使用的模型权重路径 test_weights_path，进行测试
