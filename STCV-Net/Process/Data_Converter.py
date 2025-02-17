import numpy as np
import pandas as pd
import torch
import math

def cosine_similarity(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表必须具有相同的长度")

    # 计算向量点积
    dot_product = sum(a * b for a, b in zip(list1, list2))
    
    # 计算每个向量的模长（L2范数）
    magnitude_list1 = math.sqrt(sum(a ** 2 for a in list1))
    magnitude_list2 = math.sqrt(sum(b ** 2 for b in list2))
    
    # 避免除零错误
    if magnitude_list1 == 0 or magnitude_list2 == 0:
        raise ValueError("输入向量的模长不能为0")
    
    # 计算余弦相似度
    similarity = dot_product / (magnitude_list1 * magnitude_list2)
    
    return similarity

# 最大-最小归一化函数
def Norm_MaxMin(X, maxmin=None, need_maxmin=False, need_amplitude=False):
    """
    对数据进行最大-最小归一化. 

    Args:
        X (torch.Tensor): 输入张量
        maxmin (int or list of int, optional): 归一化的最大最小值, 默认为None (全局归一化) 
        need_maxmin (bool, optional): 是否返回最小值和最大值, 默认为False
        need_amplitude (bool, optional): 是否返回振幅 (最大值 - 最小值) , 默认为False

    Returns:
        torch.Tensor or tuple: 归一化后的数据, 如果需要, 返回最小值和最大值或振幅
    """
    if maxmin is None:
        X_min = X.min().item()  # 全局最小值
        X_max = X.max().item()  # 全局最大值
    else:
        X_min = maxmin[1]  # 全局最小值
        X_max = maxmin[0]  # 全局最大值
    if need_maxmin:
        return ((X - X_min) / (X_max - X_min + 1e-9), (X_max, X_min))  # 返回归一化数据和最小最大值
    elif need_amplitude:
        return ((X - X_min) / (X_max - X_min + 1e-9), X_max - X_min)  # 返回归一化数据和振幅
    else:
        return (X - X_min) / (X_max - X_min + 1e-9)  # 返回归一化数据

def get_adjacent(adjacent_path):
    """
    从给定的文件路径读取相邻城市关系数据并返回城市列表和相邻矩阵

    Args:
        adjacent_path (str): 包含城市相邻关系的文件路径, 文件格式为 csv, 以 tab 为分隔符

    Returns:
        tuple: 
            - city_list (list): 城市名称列表
            - torch.Tensor: 转换为 Tensor 的相邻矩阵
    """
    # 读取 csv 文件, 使用 tab 分隔符, 第一行为表头, 第一列作为索引
    data = pd.read_csv(adjacent_path, sep='\t', header=0, index_col=0)
    
    # 获取列名作为城市列表
    city_list = data.columns.to_list()
    adja_matrix = torch.from_numpy(np.array(data)).unsqueeze(-1)
    
    # 将数据转换为 NumPy 数组, 再转为 Tensor 格式
    return city_list, adja_matrix

def fill_matrix(data, item_list, col_name):
    """
    根据给定的数据直接构建矩阵并转换为 Tensor

    Args:
        data (pd.DataFrame): 包含对象之间关系的数据, 包含起始对象、终点对象以及相应的关系值
        item_list (list): 对象名称列表, 作为矩阵的行列索引
        col_name (str): 要填充的列名 (从第三列开始的列)
    
    Returns:
        torch.Tensor: 填充后的对象关系矩阵, 数据类型为 Tensor
    """
    # 初始化一个 item_list 大小的零矩阵
    matrix = np.zeros((len(item_list), len(item_list)))

    # 创建 item_list 的索引字典来加速查找
    item_index = {item: idx for idx, item in enumerate(item_list)}
    
    # 遍历输入数据表的每一行
    for _, row in data.iterrows():
        start_item = row.iloc[0]  # 起始对象
        end_item = row.iloc[1]    # 目标对象
        value = row[col_name]     # 对应的关系值
        
        # 使用索引字典加速查找位置并填充矩阵
        if start_item in item_index and end_item in item_index:
            matrix[item_index[start_item], item_index[end_item]] = value

    # 直接返回转为 Tensor 的 NumPy 矩阵
    return torch.tensor(matrix, dtype=torch.float)

def get_connection(connection_path, city_list):
    """
    根据城市连接数据生成多个不同类型的连接矩阵并堆叠

    Args:
        connection_path (str): 城市连接数据文件的路径 (csv 格式)
        city_list (list): 城市名称列表
    
    Returns:
        torch.Tensor: 生成的多层次连接矩阵 (每个矩阵对应不同的列)
    """
    # 读取 csv 文件, 假设文件以 tab 为分隔符
    conn_data = pd.read_csv(connection_path, sep='\t', header=0)
    
    # 获取从第三列开始的列名列表 (这些列是不同的关系类型)
    conn_list = conn_data.columns.to_list()[2:]
    
    # 使用列表生成式直接生成所有矩阵, 然后堆叠
    matrix_list = [fill_matrix(conn_data, city_list, col) for col in conn_list]
    
    # 将生成的多个矩阵堆叠到一起, 形成三维张量
    return torch.stack(matrix_list, -1)

def get_flow(flow_path, city_list):
    """
    根据给定的城市流量数据文件生成不同时间点的流量矩阵

    Args:
        flow_path (str): 包含城市流量数据的文件路径, 文件格式为 csv, 以 tab 为分隔符
        city_list (list): 城市名称列表, 用于构建流量矩阵的行列索引
    
    Returns:
        torch.Tensor: 生成的多层次流量矩阵, 每个矩阵对应一个时间点
    """
    # 读取城市流量数据文件, 假设文件以 tab 为分隔符, 第一行为表头
    flow_data = pd.read_csv(flow_path, sep='\t', header=0)

    # 获取从第三列开始的时间列表 (这些列代表不同时间点的流量数据)
    time_list = flow_data.columns.to_list()[2:]
    # 获取数据的起始日期和终止日期
    start_date = time_list[0]
    end_date = time_list[-1]

    # 初始化一个空的列表来存储每个时间点的流量矩阵
    matrix_list = []

    # 遍历时间列表, 为每个时间点生成一个流量矩阵
    for time_str in time_list:
        matrix_list.append(fill_matrix(flow_data, city_list, time_str))

    # 将所有时间点的矩阵堆叠到一起, 形成三维张量
    return start_date, end_date, torch.stack(matrix_list, 0)

def get_randomWalk(adjacent_matrix, matrix_count=3):
    """
    定义Random_Walk 函数, 计算随机游走概率矩阵.
    
    Args:
        adjacent_matrix (torch.Tensor): 输入的邻接矩阵, 表示图的连接关系. 
        matrix_count (int): 随机游走的步数, 即计算多少次幂的概率矩阵. 默认值为 3. 

    Returns:
        torch.Tensor: 堆叠的随机游走概率矩阵, 最后一维为每次随机游走对应的概率矩阵. 
    """
    # 获取邻接矩阵的第一个元素 (通常是因为传入的是包含多个矩阵的列表或张量)
    adja = adjacent_matrix.squeeze(-1)
    
    # 归一化邻接矩阵, 使每行的元素和为 1, 形成行概率分布
    normal = adja / adja.sum(1, keepdim=True)
    
    # 计算不同步长的随机游走概率矩阵 (通过矩阵的不同次幂)
    # 使用 torch.stack 将多个步长的概率矩阵堆叠到一起
    randomWalk_matrixes = torch.stack([normal.matrix_power(i) 
                                       for i in range(1, 1 + matrix_count)], -1).float()
    return randomWalk_matrixes

def get_structure(structure_file, city_list, split_list=[0, 14, 17, 20]):
    """
    读取结构数据并基于指定的列区间计算城市间的余弦相似度矩阵。

    参数：
    - structure_file (str): 结构数据文件的路径，文件需为制表符分隔的CSV格式。
    - city_list (list): 城市名称列表，用作相似度矩阵的索引和列名。
    - split_list (list, 可选): 用于分割数据的列索引列表，默认值为 [0, 14, 17, 20]。

    返回：
    - result (list): 一个包含多个城市间余弦相似度矩阵的列表，每个矩阵为 `torch.tensor` 类型。
    """

    # 读取结构数据，使用制表符分隔符，第一行为列名，第一列作为索引
    delta_structure = pd.read_csv(structure_file, sep='\t', header=0, index_col=0)
    
    # 存放计算结果的列表
    result = []

    # 遍历 split_list 中的分段区间，分割数据并计算余弦相似度
    for i in range(1, len(split_list)):
        # 根据 split_list 中的当前区间获取数据片段
        piece_data = delta_structure.iloc[:, split_list[i - 1]:split_list[i]]
        
        # 创建一个城市间相似度矩阵，初始化为 0.0
        part_df = pd.DataFrame(index=city_list, columns=city_list).fillna(0.0)
        
        # 遍历城市列表，计算城市对之间的余弦相似度
        for city1 in city_list:
            for city2 in city_list:
                # 计算并存储城市 city1 和 city2 之间的余弦相似度
                part_df.loc[city1, city2] = cosine_similarity(piece_data.T[city1].tolist(), piece_data.T[city2].tolist())
        
        # 将计算得到的 DataFrame 转换为 torch.tensor 并添加到结果列表中
        result.append(torch.tensor(part_df.values, dtype=torch.float32))

    return torch.stack(result,-1)

def get_HFC_result(result_path, city_list):
    """
    导入HFC模型的结果.
    
    Args:
        result_path: 输入的邻接矩阵, 表示图的连接关系. 
        city_list (list): 城市名称列表, 用于构建流量矩阵的行列索引

    Returns:
        torch.Tensor: 生成的多层次矩阵, 每个矩阵对应一个时间点.
    """
    #读取数据
    df = pd.read_csv(result_path, sep = '\t', header = 0)

    # 确保 DataFrame 的列名是字符串格式
    columns = df.columns.tolist()
    
    # 获取天数（行数）和城市数量（city_list 中的唯一城市数）
    num_days = df.shape[0]
    num_cities = len(city_list)
    
    # 创建一个 tensor 来存储结果，大小为（天数，城市数，城市数）
    result_tensor = torch.zeros((num_days, num_cities, num_cities))
    
    # 遍历每一天并填充矩阵
    for day in range(num_days):
        # 初始化每天的矩阵为零矩阵
        matrix = torch.zeros((num_cities, num_cities))
        # 遍历每个城市对（起点和终点）并赋值
        for i, origin in enumerate(city_list):
            for j, destination in enumerate(city_list):
                # 生成字符串格式的城市对，如 "上海市,南京市"
                city_pair = f"{origin},{destination}"
                # 检查 city_pair 是否存在于 DataFrame 的列中
                if city_pair in columns:
                    # 将当天对应的城市对的值填入矩阵
                    matrix[i, j] = df.iloc[day][city_pair]
        # 将矩阵填入结果 tensor 中
        result_tensor[day] = matrix
        
    return result_tensor