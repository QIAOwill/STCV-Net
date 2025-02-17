import torch
import Path
from Process import Encoding
from Process import Data_Converter as Dcon

# 自定义数据集 dataset 类, 继承自 PyTorch 的 Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, Global, Edge, Flow, HFC_result,
                 num_history, future_steps, is_train):
        super().__init__()
        self.Global = Global  # 全局特征数据
        self.Edge = Edge  # 边缘特征数据
        self.Flow = Flow  # 流量数据（例如交通、流动人口等时间序列数据）
        self.HFC_result = HFC_result # HFC模型的结果
        
        self.is_train = is_train  # 是否为训练模式
        self.num_history = num_history  # 历史步数/时刻（历史窗口大小）
        self.future_steps = future_steps
        self.base = len(self.Flow) - self.num_history - (future_steps-1)  # 基础数据长度, 去除历史步数
        
    # 返回数据集长度, 训练模式下乘以复制次数
    def __len__(self):
        return self.base
    
    # 获取第 i 个样本的数据
    def __getitem__(self, i):
        # 计算实际样本的索引
        n = i % self.base + self.num_history
        # 返回特征和标签：特征包含全局、边缘特征以及历史的流量数据, 标签为当前时刻的流量数据
        return ((self.Global[n], self.Edge, self.Flow[n - self.num_history : n]), 
                self.Flow[n : n + self.future_steps], 
                self.HFC_result[n : n + self.future_steps])
    
def Read_file(pre_impact, slope, normal = True, Flow_only = False):
    """
    根据数据路径对数据进行读取并归一化
    Args:
        pre_impact (int): 假期前影响天数;
        slope (float): 控制假期前影响的坡度参数;
        normal (bool): 是否进行归一化, 默认为 True;
        Flow_only (bool): 是否只读取流量数据, 默认为 False;
    
    Returns:
        tuple:
            - Global (torch.tensor): 全局时间编码数据;
            - Edge (torch.tensor): 网络静态特征信息;
            - Flow (torch.tensor): 流量信息;
            - max_min (float): 归一化过程中的极值信息;
    """
    if Flow_only:
        # 获取邻接矩阵, 并归一化邻接矩阵
        print('*'*100,'\n读取数据中... ...\n')
        city_list, adjacent_matrix = Dcon.get_adjacent(Path.adjacent_file)
        # 读取流量数据
        start_date, end_date, Flow_matrixes = Dcon.get_flow(Path.flow_file, city_list)
            # 判断是否需要将数值进行归一化
        if normal:
            (Flow, max_min) = Dcon.Norm_MaxMin(Flow_matrixes, need_maxmin = True)
        else:
            # 不归一化, 则用原来的值
            Flow = Flow_matrixes
            max_min = None
        print('*'*100,'\n读取完成！\n')
        return Flow, max_min
    else:
        print('*'*100,'\n读取数据中... ...\n')
        # 获取邻接矩阵, 并归一化邻接矩阵
        city_list, adjacent_matrix = Dcon.get_adjacent(Path.adjacent_file)
        # 获取连接信息, 并归一化的连接信息矩阵
        conn_matrixes = Dcon.get_connection(Path.connection_file, city_list)
        # 获取随机游走矩阵, 并进行归一化处理
        randomWalk_matrixes = Dcon.get_randomWalk(adjacent_matrix, matrix_count=3)
        # 获取结构数据
        structure = Dcon.get_structure(Path.structure_file, city_list)
        # 获取 Flow_, 即归一化的 OD 流量数据 (Origin-Destination Flow)
        start_date, end_date, Flow_matrixes = Dcon.get_flow(Path.flow_file, city_list)
        # 获取ENGM模型的运行结果
        HFC_result = Dcon.get_HFC_result(Path.HFC_result_file, city_list)

        # 判断是否需要将数值进行归一化
        if normal:
            AM,CM,RW = Dcon.Norm_MaxMin(adjacent_matrix), Dcon.Norm_MaxMin(conn_matrixes), Dcon.Norm_MaxMin(randomWalk_matrixes)
            (Flow, max_min) = Dcon.Norm_MaxMin(Flow_matrixes, need_maxmin = True)
            HFC_result = Dcon.Norm_MaxMin(HFC_result, maxmin = max_min)
        else:
            # 不归一化, 则用原来的值
            AM,CM,RW = adjacent_matrix, conn_matrixes, randomWalk_matrixes
            Flow = Flow_matrixes
            max_min = None

        TE = Encoding.Temporal_Encoding(start_date, end_date, pre_impact, slope)

        Global=torch.concatenate([TE],-1)
        Edge=torch.concatenate([AM,CM,RW,structure],-1)
        
        print('*'*100,'\n读取完成！\n')
        return (Global, Edge, Flow, 
                HFC_result, max_min)

def Split_data(Global, Edge, Flow, HFC_result, 
               num_history, future_steps, split_bound):
    """
    根据读取的数据, 进行数据集的拆分

    Args:
        Global (torch.tensor): 全局时间编码数据;
        Edge (torch.tensor): 网络静态特征信息;
        Flow (torch.tensor): 流量信息;
        HFC_result (torch.tensor): HFC模型的结果;
        num_history (int): 历史数据长度;
        future_steps (int): 未来预测步长;
        split_bound (list): 数据划分;
    
    Returns:
        tuple:
            - dataset (torch.tensor): 数据集, 包含训练集、验证集和测试集;
            - dim_time_in (torch.tensor): 时间的初始维度;
            - dim_Edge_in (torch.tensor): 特征的初始维度;
    """
    # 定义 dataset 数据集, 包含训练集、验证集和测试集
    dataset = {
        # 训练集：使用 Global、Edge 和 Flow 数据的训练部分, 并将 is_train 参数设置为 True
        'train': Dataset(
            Global[:split_bound[0]], Edge, Flow[:split_bound[0]], 
            HFC_result[:split_bound[0]],
            num_history, future_steps, True  # 标记为训练模式
        ),
        
        # 验证集：使用 Global、Edge 和 Flow 数据的验证部分, is_train 设置为 False
        'validate': Dataset(
            Global[split_bound[0]:split_bound[1]], Edge, Flow[split_bound[0]:split_bound[1]],
            HFC_result[split_bound[0]:split_bound[1]], 
            num_history, future_steps, False  # 标记为非训练模式
        ),
        
        # 测试集：使用 Global、Edge 和 Flow 数据的测试部分, is_train 设置为 False
        'test': Dataset(
            Global[split_bound[1]:], Edge, Flow[split_bound[1]:], 
            HFC_result[split_bound[1]:], 
            num_history, future_steps, False  # 标记为非训练模式
        )
    }
    # 获取时间和特征的初始维度
    dim_time_in=dataset['train'][0][0][0].shape[-1]
    dim_Edge_in=dataset['train'][0][0][1].shape[-1]

    return dataset, dim_time_in, dim_Edge_in