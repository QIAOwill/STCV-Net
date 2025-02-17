import Path
import torch
from torch.utils.data import DataLoader
import os
from Process.Loss import Format_Metrics

def set_seed(seed = 54321):
    # 设置 PyTorch 的 CPU 和 GPU 随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况下需要设置所有 GPU 的种子

def denormalize_tensor(tensor, max_min):
    """
    将归一化的 tensor 还原为原始值。
    
    参数:
        tensor (torch.Tensor): 归一化的 tensor。
        max_min (tuple): 原始数据的最大值和最小值 (max, min)。
        
    返回:
        torch.Tensor: 还原后的 tensor。
    """
    max_value, min_value = max_min
    return tensor * (max_value - min_value) + min_value

# 定义训练函数 Train
def Train(model, optimizer, scaler, dataloader, loss_fn, metrics_fn, device):
    """
    训练模型的单次迭代。
    
    Args:
        model (torch.nn.Module): 要训练的模型
        optimizer (torch.optim.Optimizer): 优化器, 用于更新模型参数
        scaler (torch.cuda.amp.GradScaler): 用于自动混合精度训练
        dataloader (DataLoader): 训练数据加载器
        loss_fn (function): 损失函数
        metrics_fn (function): 用于评估模型性能的指标函数
        device (str): 训练设备
    
    返回:
    list: 每次训练后累积的性能指标
    """
    model.train()  # 设置模型为训练模式
    n = len(dataloader)  # 获取数据集的总批次数
    metrics = []  # 存储训练中的评估指标
    
    # 遍历数据集进行训练
    for (batch, data) in enumerate(dataloader):
        (input, target, HFC) = data  # 获取输入和目标数据
        with torch.autocast(device_type = device, dtype=torch.float16):  # 使用自动混合精度
            predict = model([x.to(device) for x in input], 
                            HFC.to(device))  # 模型前向传播
            target = target.permute(0,2,3,1).to(device)  # 将目标数据移至计算设备
            loss = loss_fn(predict, target)  # 计算损失
  
        scaler.scale(loss).backward()  # 使用GradScaler进行反向传播
        
        # 梯度裁剪
        max_norm = 2.0  # 设定最大梯度范数
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)  # 更新模型参数
        scaler.update()  # 更新Scaler
        optimizer.zero_grad(set_to_none=True)  # 梯度清零
        # 计算并存储当前批次的指标
        metrics.append(metrics_fn(predict, target))  

    return [sum(x) / n for x in zip(*metrics)]  # 返回平均指标

# 定义验证函数 Validate
def Validate(model, dataloader, metrics_fn, device):
    """
    测试模型性能。
    
    Args:
        model (torch.nn.Module): 要测试的模型
        dataloader (DataLoader): 测试数据加载器
        metrics_fn (function): 用于评估模型性能的指标函数
        device (str): 训练设备
    
    返回:
    list: 每次测试后的累积性能指标
    """
    model.eval()  # 设置模型为评估模式
    n = len(dataloader)  # 获取数据集的总批次数
    metrics = []  # 存储测试中的评估指标

    with torch.inference_mode():  # 关闭梯度计算, 节省内存和加速推理
        for data in dataloader:
            (input, target, HFC) = data  # 获取输入和目标数据
            predict = model([x.to(device) for x in input], 
                            HFC.to(device))  # 模型前向传播
            target = target.permute(0,2,3,1).to(device)  # 将目标数据移至计算设备
            metrics.append(metrics_fn(predict.where(predict>0,0), target))  # 计算并存储当前批次的指标
    
    return [sum(x) / n for x in zip(*metrics)]  # 返回平均指标

# 定义测试函数 Test
def Test(model, dataloader, metrics_fn, device):
    """
    测试模型性能。
    
    Args:
        model (torch.nn.Module): 要测试的模型
        dataloader (DataLoader): 测试数据加载器
        metrics_fn (function): 用于评估模型性能的指标函数
        device (str): 训练设备
    
    返回:
    list: 每次测试后的累积性能指标
    """
    model.eval()  # 设置模型为评估模式
    n = len(dataloader)  # 获取数据集的总批次数
    metrics = []  # 存储测试中的评估指标

    with torch.inference_mode():  # 关闭梯度计算, 节省内存和加速推理
        for data in dataloader:
            (input, target, HFC) = data  # 获取输入和目标数据
            predict = model([x.to(device) for x in input], 
                            HFC.to(device))  # 模型前向传播
            target = target.permute(0,2,3,1).to(device)  # 将目标数据移至计算设备
            metrics.append(metrics_fn(predict.where(predict>0,0), target))  # 计算并存储当前批次的指标
    
    return predict, [sum(x) / n for x in zip(*metrics)]  # 返回平均指标


# 定义训练循环函数 Loop
def Loop(model, para, optimizer, scheduler, scaler, dataset, loss_fn, metrics_fn,
         batch_size=32, num_epochs=100, num_workers=0, device = 'cuda', max_min = (143350.0, 0.0)):
    """
    完整的训练循环, 包括训练和验证, 支持加载检查点。
    
    Args:
        model (torch.nn.Module): 要训练的模型
        para : 参数类
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler): 学习率调度器
        scaler (torch.cuda.amp.GradScaler): 用于自动混合精度训练
        dataset (dict): 包含训练、验证和测试数据集的字典
        loss_fn (function): 损失函数
        metrics_fn (function): 指标函数
        batch_size (int): 每个批次的样本数, 默认32
        num_epochs (int): 训练的总轮数, 默认100
        num_workers (int): 数据加载时的工作进程数, 默认0
        gap (int): 用于控制每次输出的步长, 默认100
        load_checkpoint (bool): 是否加载检查点, 默认False
        device (str): 训练设备, 默认为 cuda
    """
    set_seed()
    print('='*100)
    # 创建数据加载器字典
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size, pin_memory=True, 
                            num_workers=num_workers, shuffle=True),
        'validate': DataLoader(dataset['validate'], batch_size, pin_memory=True, 
                               num_workers=num_workers),
        'test': DataLoader(dataset['test'], batch_size, pin_memory=True, 
                           num_workers=num_workers)
    }
    model = model.to(device)  # 将模型移至计算设备
    path = Path.update_path(Path.result_path) # 设置模型保存路径
    if not os.path.exists(path):  # 如果路径不存在则创建
        os.makedirs(path)
    start_epoch = 1  # 初始化起始轮数
       
    # 开始训练循环
    for epoch in range(start_epoch, 1 + num_epochs):
        print(f'Epoch: {epoch:0>3d}/{num_epochs:0>3d}   '+
              f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        metrics = {
            'train': Train(model, optimizer, scaler, dataloader['train'], 
                           loss_fn, metrics_fn, device),  # 训练
            'validate': Validate(model, dataloader['validate'], metrics_fn, device),  # 验证
            'test': Test(model, dataloader['test'], metrics_fn, device)  # 测试
        }
        scheduler.step()  # 更新学习率调度器
        if metrics["test"][1][4] < para.save_point:
            # 保存当前轮次的模型检查点
            torch.save(denormalize_tensor(metrics["test"][0].squeeze(0).permute(2, 0, 1), max_min), 
                       path + ('/' + f'Epoch {epoch:0>3d}.pt'))

        # 输出训练、验证和测试的损失信息
        print('-'*150,'\n',
              f'  Train Loss:  {Format_Metrics(metrics["train"])}\n' +
              f'Validate Loss:  {Format_Metrics(metrics["validate"])}\n' +
              f'    Test Loss:  {Format_Metrics(metrics["test"][1])}\n',
              '-'*150,'\n\n',
            )