import math

import numpy as np
import pandas as pd
import torch


def cosine_similarity(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表必须具有相同的长度")

    dot_product = sum(a * b for a, b in zip(list1, list2))
    magnitude_list1 = math.sqrt(sum(a ** 2 for a in list1))
    magnitude_list2 = math.sqrt(sum(b ** 2 for b in list2))
    if magnitude_list1 == 0 or magnitude_list2 == 0:
        raise ValueError("输入向量的模长不能为0")
    return dot_product / (magnitude_list1 * magnitude_list2)


# 最大-最小归一化函数
def Norm_MaxMin(X, maxmin=None, need_maxmin=False, need_amplitude=False):
    """对 Tensor 进行全局 Min-Max 归一化。"""
    if maxmin is None:
        X_min = X.min().item()
        X_max = X.max().item()
    else:
        X_min = maxmin[1]
        X_max = maxmin[0]

    normalized = (X - X_min) / (X_max - X_min + 1e-9)
    if need_maxmin:
        return normalized, (X_max, X_min)
    if need_amplitude:
        return normalized, X_max - X_min
    return normalized


def get_adjacent(adjacent_path):
    """读取邻接矩阵并返回城市列表与 float32 Tensor。"""
    data = pd.read_csv(adjacent_path, sep='\t', header=0, index_col=0)
    city_list = data.columns.to_list()
    values = data.to_numpy(dtype=np.float32, copy=True)
    return city_list, torch.from_numpy(values).unsqueeze(-1)


def _pair_indices(data, item_list):
    """将关系表前两列的起终点名称一次性转换为矩阵下标。"""
    item_index = {item: idx for idx, item in enumerate(item_list)}
    origin = data.iloc[:, 0].map(item_index)
    destination = data.iloc[:, 1].map(item_index)
    valid = origin.notna() & destination.notna()
    return (
        origin[valid].astype(np.int64).to_numpy(),
        destination[valid].astype(np.int64).to_numpy(),
        valid.to_numpy(),
    )


def fill_matrix(data, item_list, col_name):
    """根据关系表构建单个矩阵；保留旧接口，但使用向量化赋值。"""
    n = len(item_list)
    matrix = np.zeros((n, n), dtype=np.float32)
    origin_idx, destination_idx, valid = _pair_indices(data, item_list)
    values = pd.to_numeric(data.loc[valid, col_name], errors='raise').to_numpy(dtype=np.float32)
    matrix[origin_idx, destination_idx] = values
    return torch.from_numpy(matrix)


def get_connection(connection_path, city_list):
    """读取全部连接特征；一次性矩阵化，避免每列重复 iterrows。"""
    conn_data = pd.read_csv(connection_path, sep='\t', header=0)
    feature_cols = conn_data.columns.to_list()[2:]
    n = len(city_list)
    result = np.zeros((n, n, len(feature_cols)), dtype=np.float32)

    origin_idx, destination_idx, valid = _pair_indices(conn_data, city_list)
    values = conn_data.loc[valid, feature_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    result[origin_idx, destination_idx, :] = values
    return torch.from_numpy(result)


def get_flow(flow_path, city_list):
    """读取 OD 流量并一次性转换为 [day, origin, destination] Tensor。"""
    flow_data = pd.read_csv(flow_path, sep='\t', header=0)
    time_list = flow_data.columns.to_list()[2:]
    if not time_list:
        raise ValueError(f"流量文件没有时间列: {flow_path}")

    start_date, end_date = time_list[0], time_list[-1]
    n = len(city_list)
    result = np.zeros((len(time_list), n, n), dtype=np.float32)

    origin_idx, destination_idx, valid = _pair_indices(flow_data, city_list)
    values = flow_data.loc[valid, time_list].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    # values: [OD, day] -> result: [day, O, D]
    result[:, origin_idx, destination_idx] = values.T
    return start_date, end_date, torch.from_numpy(result)


def get_randomWalk(adjacent_matrix, matrix_count=3):
    """计算 1..matrix_count 阶随机游走矩阵。"""
    adja = adjacent_matrix.squeeze(-1).float()
    row_sum = adja.sum(1, keepdim=True)
    # 对完全孤立的节点避免除零；其随机游走行保持为 0。
    normal = torch.where(row_sum > 0, adja / row_sum.clamp_min(1e-12), torch.zeros_like(adja))
    return torch.stack([normal.matrix_power(i) for i in range(1, 1 + matrix_count)], -1).float()


def get_structure(structure_file, city_list, split_list=(0, 14, 17, 20)):
    """
    读取城市结构特征并计算分组余弦相似度。

    原实现对每个城市对逐一调用 Python 函数；这里改为矩阵乘法，结果等价但读取/预处理快很多。
    """
    delta_structure = pd.read_csv(structure_file, sep='\t', header=0, index_col=0)
    missing = [city for city in city_list if city not in delta_structure.index]
    if missing:
        raise KeyError(f"结构数据缺少城市: {missing}")

    delta_structure = delta_structure.loc[city_list]
    result = []
    for start, end in zip(split_list[:-1], split_list[1:]):
        piece = delta_structure.iloc[:, start:end].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
        norms = np.linalg.norm(piece, axis=1, keepdims=True)
        if np.any(norms <= 0):
            bad = [city_list[i] for i in np.where(norms[:, 0] <= 0)[0]]
            raise ValueError(f"结构特征模长为0，无法计算余弦相似度: {bad}")
        similarity = (piece @ piece.T) / (norms @ norms.T)
        result.append(torch.from_numpy(similarity.astype(np.float32, copy=False)))

    return torch.stack(result, -1)


def get_HFC_result(result_path, city_list):
    """读取 HFC 结果并转换为 [day, origin, destination] Tensor。"""
    df = pd.read_csv(result_path, sep='\t', header=0)
    n = len(city_list)
    result = np.zeros((len(df), n, n), dtype=np.float32)
    item_index = {city: idx for idx, city in enumerate(city_list)}

    # 每一列对应一个 OD 对；整列一次赋值，避免 day × city × city 三重 Python 循环。
    for col in df.columns:
        if ',' not in str(col):
            continue
        origin, destination = str(col).split(',', 1)
        if origin in item_index and destination in item_index:
            values = pd.to_numeric(df[col], errors='raise').to_numpy(dtype=np.float32)
            result[:, item_index[origin], item_index[destination]] = values

    return torch.from_numpy(result)
