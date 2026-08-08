import pandas as pd
import torch

import Path
from Process import Encoding
from Process import Data_Converter as Dcon


class Dataset(torch.utils.data.Dataset):
    def __init__(self, Global, Edge, Flow, HFC_result,
                 num_history, future_steps, is_train):
        super().__init__()
        self.Global = Global
        self.Edge = Edge
        self.Flow = Flow
        self.HFC_result = HFC_result
        self.is_train = is_train
        self.num_history = num_history
        self.future_steps = future_steps
        self.base = len(self.Flow) - self.num_history - (future_steps - 1)
        if self.base <= 0:
            raise ValueError(
                f"数据长度不足: len(Flow)={len(self.Flow)}, num_history={num_history}, "
                f"future_steps={future_steps}"
            )

    def __len__(self):
        return self.base

    def __getitem__(self, i):
        n = i % self.base + self.num_history
        return (
            (self.Global[n], self.Edge, self.Flow[n - self.num_history:n]),
            self.Flow[n:n + self.future_steps],
            self.HFC_result[n:n + self.future_steps],
        )


class DataCache:
    """
    单进程内的数据缓存。

    1. 邻接/连接/结构/OD/HFC 文件只从磁盘读取一次；
    2. 时间编码按 (pre_impact, slope) 缓存；
    3. Flow/HFC 归一化结果按 (train_ratio, future_steps) 缓存；
    4. 因此网格搜索只改变 batch_size、网络维度等模型参数时，不再重复读文件。
    """
    def __init__(self):
        self._raw = None
        self._temporal_cache = {}
        self._normalized_cache = {}
        self._edge_normalized = None
        self._prepared_cache = {}

    @property
    def city_list(self):
        self._ensure_raw()
        return list(self._raw['city_list'])

    @property
    def start_date(self):
        self._ensure_raw()
        return self._raw['start_date']

    @property
    def end_date(self):
        self._ensure_raw()
        return self._raw['end_date']

    def _ensure_raw(self):
        if self._raw is not None:
            return

        print('*' * 100)
        print('首次读取数据中...（本次进程后续参数组合将直接复用内存缓存）')

        city_list, adjacent_matrix = Dcon.get_adjacent(Path.adjacent_file)
        conn_matrixes = Dcon.get_connection(Path.connection_file, city_list)
        random_walk = Dcon.get_randomWalk(adjacent_matrix, matrix_count=3)
        structure = Dcon.get_structure(Path.structure_file, city_list)
        start_date, end_date, flow_matrixes = Dcon.get_flow(Path.flow_file, city_list)
        hfc_result = Dcon.get_HFC_result(Path.HFC_result_file, city_list)

        if len(flow_matrixes) != len(hfc_result):
            raise ValueError(
                f"Flow 与 HFC 天数不一致: Flow={len(flow_matrixes)}, HFC={len(hfc_result)}"
            )

        expected_days = len(pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date)))
        if expected_days != len(flow_matrixes):
            raise ValueError(
                f"OD 数据时间列数量({len(flow_matrixes)})与日期范围 {start_date}~{end_date} "
                f"对应天数({expected_days})不一致。"
            )

        self._raw = dict(
            city_list=city_list,
            adjacent=adjacent_matrix.float(),
            connection=conn_matrixes.float(),
            random_walk=random_walk.float(),
            structure=structure.float(),
            start_date=start_date,
            end_date=end_date,
            flow=flow_matrixes.float(),
            hfc=hfc_result.float(),
        )
        print('首次读取完成，原始数据已缓存。')
        print('*' * 100)

    def _get_temporal(self, pre_impact, slope):
        key = (int(pre_impact), float(slope))
        if key not in self._temporal_cache:
            self._temporal_cache[key] = Encoding.Temporal_Encoding(
                self._raw['start_date'], self._raw['end_date'], pre_impact, slope
            ).float()
        return self._temporal_cache[key]

    def _get_edge(self, normal):
        if not normal:
            return torch.concatenate([
                self._raw['adjacent'],
                self._raw['connection'],
                self._raw['random_walk'],
                self._raw['structure'],
            ], -1)

        if self._edge_normalized is None:
            am = Dcon.Norm_MaxMin(self._raw['adjacent'])
            cm = Dcon.Norm_MaxMin(self._raw['connection'])
            rw = Dcon.Norm_MaxMin(self._raw['random_walk'])
            self._edge_normalized = torch.concatenate([am, cm, rw, self._raw['structure']], -1).float()
        return self._edge_normalized

    def _get_flow_hfc(self, normal, train_ratio, future_steps):
        if not normal:
            return self._raw['flow'], self._raw['hfc'], None

        key = (float(train_ratio), int(future_steps))
        if key not in self._normalized_cache:
            test_start = self._raw['flow'].shape[0] - int(future_steps)
            train_end = int(test_start * float(train_ratio))
            if train_end <= 0 or test_start <= train_end:
                raise ValueError(
                    f"无效的数据划分: train_ratio={train_ratio}, future_steps={future_steps}, "
                    f"总天数={self._raw['flow'].shape[0]}"
                )
            _, max_min = Dcon.Norm_MaxMin(self._raw['flow'][:train_end], need_maxmin=True)
            flow = Dcon.Norm_MaxMin(self._raw['flow'], maxmin=max_min).float()
            hfc = Dcon.Norm_MaxMin(self._raw['hfc'], maxmin=max_min).float()
            self._normalized_cache[key] = (flow, hfc, max_min)
        return self._normalized_cache[key]

    def get(self, pre_impact, slope, normal=True, Flow_only=False,
            train_ratio=0.8, future_steps=8):
        self._ensure_raw()

        if Flow_only:
            flow, _, max_min = self._get_flow_hfc(normal, train_ratio, future_steps)
            return flow, max_min

        key = (
            int(pre_impact), float(slope), bool(normal),
            float(train_ratio), int(future_steps)
        )
        if key in self._prepared_cache:
            print(f'数据缓存命中: {key}，跳过磁盘读取与重复预处理。')
            return self._prepared_cache[key]

        temporal = self._get_temporal(pre_impact, slope)
        edge = self._get_edge(normal)
        flow, hfc, max_min = self._get_flow_hfc(normal, train_ratio, future_steps)
        global_features = temporal

        prepared = (global_features, edge, flow, hfc, max_min)
        self._prepared_cache[key] = prepared
        return prepared

    def test_dates(self, future_steps):
        self._ensure_raw()
        dates = pd.date_range(start=pd.to_datetime(self._raw['start_date']), end=pd.to_datetime(self._raw['end_date']))
        return [d.strftime('%Y-%m-%d') for d in dates[-int(future_steps):]]


_DEFAULT_CACHE = DataCache()


def clear_cache():
    """需要在同一 Python 进程中修改数据文件后重新读取时调用。"""
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = DataCache()


def Read_file(pre_impact, slope, normal=True, Flow_only=False,
              train_ratio=0.8, future_steps=8, cache=None):
    """
    兼容原接口的数据读取函数。默认使用进程级缓存；也可从 Main.py 显式传入 DataCache。
    """
    cache = _DEFAULT_CACHE if cache is None else cache
    return cache.get(
        pre_impact, slope, normal=normal, Flow_only=Flow_only,
        train_ratio=train_ratio, future_steps=future_steps
    )


def Split_data(Global, Edge, Flow, HFC_result,
               num_history, future_steps, split_bound):
    """按照训练/验证/测试目标边界构建 Dataset。"""
    train_end, test_start = split_bound
    validate_start = train_end - num_history
    test_context_start = test_start - num_history

    if validate_start < 0 or test_context_start < 0:
        raise ValueError(
            f"num_history={num_history} 过大，无法为验证/测试集保留历史窗口。"
        )

    dataset = {
        'train': Dataset(
            Global[:train_end], Edge, Flow[:train_end], HFC_result[:train_end],
            num_history, future_steps, True
        ),
        'validate': Dataset(
            Global[validate_start:test_start], Edge, Flow[validate_start:test_start],
            HFC_result[validate_start:test_start], num_history, future_steps, False
        ),
        'test': Dataset(
            Global[test_context_start:], Edge, Flow[test_context_start:],
            HFC_result[test_context_start:], num_history, future_steps, False
        ),
    }

    dim_time_in = dataset['train'][0][0][0].shape[-1]
    dim_edge_in = dataset['train'][0][0][1].shape[-1]
    return dataset, dim_time_in, dim_edge_in
