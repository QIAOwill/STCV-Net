import torch
import Path
from Process import Paras
from Process import Data_Loader as DL
from Model.HFC import HFC_Model
from Process import Learning, Loss
from Process.Time import time, output_duration

param_dict = dict(
        num_history = [14,7],             # 历史数据长度
        future_steps = [8],             # 未来预测步长
        batch_size = [8,16,32],              # 训练批次数
        split_bound = [100],            # 数据划分界限

        dim_time = [8],                 # 时间的编码维度
        gcn_list = [(64, 256, 64)],     # 图卷积的隐藏维度
        dim_edge = [16],                # 特征的编码维度
        dim_lstm = [128],               # 时间注意力中 LSTM 模型的隐藏层维度
        num_heads = [4],                # 多头注意力数量
        num_blocks = [4],               # 块的数量
        map_layers = [(64, 256, 64)],   # 输出映射的隐藏层     
        pre_impact = [2],               # 假期前影响天数
        slope = [0.2],                  # 控制假期前影响的坡度参数;
        num_epochs = [100],             # 训练次数 
        num_workers = [0],              # 设置加载数据时使用的线程数量，0 表示使用主线程加载
        
        device = ['cuda'],              # 运行设备信息
        num_cities = [''],              # 城市数量
        dim_time_in = [''],             # 时间的输入维度
        dim_edge_in = [''],             # 特征的输入维度
        save_point = [40],              # 是否储存每次训练的文件
        seed = [54321],                 # 随机种子
        start_time = [time.time()],     # 初始化起始时间
        end_time = [time.time()],       # 初始化终止时间
    )

Model_part = ['HFC', 'Main_Model'][1]

if __name__ == '__main__': 
    if Model_part == 'HFC':
        namelist = ['Prophet']
        precict_day = 8
        citys_range = None

        HFC_Model(  namelist[0], 
                    precict_day = precict_day, 
                    citys_range = citys_range,
                    data_path = Path.root_file + '/data.txt',
                    predict_path = Path.root_file + '/predict.txt',
                    od_path = Path.root_file + '/delta OD.txt')

    if Model_part == 'Main_Model':
        from Model import Main_Model
        #获取程序开始时间
        start_time = time.time()
        #获取程序开始时间
        start_time = time.time()
        parameters_list = Paras.list_of_param_dicts(param_dict)
        for parameters in parameters_list:
            # 获取参数类
            para = Paras.Args(parameters)
            para.start_time = time.time() # 每次循环重置时间
            # 读取数据
            (Global, Edge, Flow, HFC_result, max_min) = DL.Read_file(para.pre_impact, para.slope)
            para.num_cities = Flow.shape[-1]
            # 设置验证集和测试集之间的界限
            para.split_bound = (para.split_bound, 
                                Flow.shape[0] - para.num_history - para.future_steps)
            # 分割数据
            dataset, dim_time_in, dim_edge_in = DL.Split_data(  Global, Edge, Flow, 
                                                                HFC_result,
                                                                para.num_history, 
                                                                para.future_steps, 
                                                                para.split_bound)
            # 设置参数
            para.dim_time_in = dim_time_in
            para.dim_edge_in = dim_edge_in
            # 初始化 Main_Model 模型
            model = Main_Model.Main_Model(para)
            # 初始化 Adam 优化器
            optimizer = torch.optim.Adam(model.parameters(), 3e-4)
            # 初始化学习率调度器，使用指数学习率衰减，每次迭代后学习率乘以 0.99
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            # 初始化 GradScaler，用于自动混合精度训练（AMP），以提高训练效率
            scaler = torch.cuda.amp.GradScaler()
            # 调用训练循环函数，进行模型的训练和验证
            Learning.Loop(
                model,                    # 传入模型
                para,                     # 传入参数类
                optimizer,                # 传入优化器
                scheduler,                # 传入学习率调度器
                scaler,                   # 传入自动混合精度缩放器
                dataset,                  # 传入数据集（包含训练、验证、测试集）
                torch.nn.MSELoss(),       # 使用均方误差损失函数
                Loss.Metrics(max_min),    # 自定义的损失度量，用 max_min 衡量
                batch_size = para.batch_size,           # 训练批次数
                num_epochs = para.num_epochs,           # 训练次数
                num_workers = para.num_workers,         # 设置加载数据时使用的线程数量，0 表示使用主线程加载
                device = para.device,                   # 训练设备
                max_min = max_min         # 归一化前的最大最小值
            )
        # 获取算法结束时间
        end_time = time.time()
        # 计算并输出运行时间
        _, _, _ = output_duration(start_time, end_time, print_time = True)