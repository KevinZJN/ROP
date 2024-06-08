import  math
import  torch

#定义了一个函数 create_lr_scheduler，用于创建学习率调度器（lr_scheduler）
def create_lr_scheduler(optimizer,       #优化器对象，用于更新模型的参数
                        num_step: int,    #每个 epoch 中的迭代步数
                        epochs: int,      #总的训练 epoch 数量
                        warmup=True,      #是否进行学习率预热，默认为 True
                        warmup_epochs=1,  #学习率预热的 epoch 数量，默认为 1
                        warmup_factor=1e-3, #学习率预热阶段的初始学习率倍率因子，默认为 1e-3
                        end_factor=1e-6      #训练结束时的学习率倍率因子，默认为 1e-6
                        ):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):     #定义了一个内嵌函数 f(x)，用于根据当前的迭代步数 x 返回学习率倍率因子。在训练开始之前，PyTorch
                  # 会提前调用一次 lr_scheduler.step() 方法，因此 f(x) 中需要考虑预热阶段的学习率倍率因子
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):   #如果 warmup 为 True 且当前迭代步数 x 小于等于预热阶段的总步数（warmup_epochs * num_step）
            alpha = float(x) / (warmup_epochs * num_step)        #，则采用线性插值的方式计算学习率倍率因子，从 warmup_factor 逐渐增加到 1
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:                                              #否则，当前步数大于预热阶段的总步数，采用余弦退火的方式计算学习率倍率因子，从 1 逐渐减小到 end_factor
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)    #函数返回一个使用自定义学习率倍率因子 f 的 LambdaLR 调度器对象，通过 torch.optim.lr_scheduler.LambdaLR 创建