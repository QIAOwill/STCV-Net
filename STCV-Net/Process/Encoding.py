import torch
import numpy as np
import pandas as pd

holiday_list = [
    (pd.Timestamp('2022-12-31'), pd.Timestamp('2023-01-02')),  # 元旦假期
    (pd.Timestamp('2023-01-21'), pd.Timestamp('2023-01-27')),  # 春节假期
    (pd.Timestamp('2023-04-05'), pd.Timestamp('2023-04-05')),  # 清明节假期
    (pd.Timestamp('2023-04-29'), pd.Timestamp('2023-05-03')),  # 劳动节假期
    (pd.Timestamp('2023-06-22'), pd.Timestamp('2023-06-24')),  # 端午节假期
    (pd.Timestamp('2023-09-29'), pd.Timestamp('2023-10-06')),  # 中秋节 & 国庆节假期
]

def Temporal_Encoding(start_date, end_date, pre_impact, k, holiday_list = holiday_list):
    """
    计算每一天的时间编码, 输出大小为 daynum*6 的 torch.tensor. 
    
    Args:
        start_date (str): 起始日期, 格式为 'YYYYMMDD'. 
        end_date (str): 终止日期, 格式为 'YYYYMMDD'. 
        pre_impact (int): 假期前影响的总天数. 
        k (float): 控制假期前影响的坡度参数. 
        holiday_list (list of tuples): 假期列表, 每个假期由一个元组表示, 元组包含假期开始日期和结束日期. 
    
    Returns:
        torch.tensor: 大小为 daynum*6 的 tensor, 每一行包含六个时间编码特征. 
    """
    
    # 创建从起始日期到终止日期的日期范围
    date_range = pd.date_range(start=start_date, end=end_date)

    # 初始化空列表, 用于存储每一天的时间编码
    encodings = []

    # 定义 holiday length (HL)函数 HL_factor(HL) = log(1 + HL)
    def HL_factor(HL):
        return np.log(1 + HL)

    # 定义函数 pre-holiday_factor(d, N), 用于计算假期前的影响
    def PH_factor(d_num, pre_impact):
        return 1 / (1 + np.exp(k * (d_num - pre_impact / 2)))

    # 遍历每一天, 计算相应的时间编码
    for current_date in date_range:
        # 计算 p_week, 其中 r_week 是当前日期在一周中的位置 (0 表示星期一, 6 表示星期天)
        r_week = current_date.weekday()  # 获取当前日期是一周中的第几天
        r_week_max = 7  # r_week 的最大值为 6, 表示一周有 7 天
        p_week = (r_week+1) / r_week_max  # 计算 p_week

        # 计算 cos(2π • pweek) 和 sin(2π • pweek)
        cos_pweek = np.cos(2 * np.pi * p_week)
        sin_pweek = np.sin(2 * np.pi * p_week)

        # 假期相关计算
        is_holiday = False  # 标记当前日期是否在假期内
        p_holiday = 0  # p_holiday 表示当前日期在假期中的位置, 初始为 0
        HL = 0  # 假期长度 (HL), 初始为 0
        d_num = float('inf')  # 距离最近假期还有多少天, 初始为无穷大
        
        # 遍历假期列表, 检查当前日期是否在假期内或距离假期有多远
        for holiday_start, holiday_end in holiday_list:
            # 如果当前日期在某个假期范围内
            if holiday_start <= current_date <= holiday_end:
                is_holiday = True
                # 计算假期长度 holiday length (HL)
                HL = (holiday_end - holiday_start).days + 1
                # 计算当前日期在假期中的位置 p_holiday
                holiday_no = (current_date - holiday_start).days + 1
                p_holiday = holiday_no / HL
                break  # 一旦找到假期, 直接退出循环, 因为只计算最近的假期

            # 如果当前日期在假期之前, 计算距离假期还有多少天 (d)
            if current_date < holiday_start:
                d_num = min(d_num, (holiday_start - current_date).days)

        # 计算 cos(2π • pholiday) 和 sin(2π • pholiday)
        cos_pholiday = np.cos(2 * np.pi * p_holiday)
        sin_pholiday = np.sin(2 * np.pi * p_holiday)

        # f_hl 和 f_ph 只在满足条件时计算, 否则为 0
        if is_holiday:
            f_hl = HL_factor(HL)  # 如果当前日期在假期内, 计算 f(HL)
        else:
            f_hl = 0  # 否则 f_hl 为 0

        if d_num <= pre_impact:  # 如果距离假期的天数小于等于 N, 计算 g(d, N)
            f_ph = PH_factor(d_num, pre_impact)
        elif is_holiday:
            f_ph = 1
        else:
            f_ph = 0  # 否则 f_ph 为 0

        # 将计算的时间编码添加到列表中
        encoding = [cos_pweek, sin_pweek, cos_pholiday, sin_pholiday, f_hl, f_ph]
        encodings.append(encoding)

    # 将列表转换为 torch tensor, 并返回
    encodings_tensor = torch.tensor(encodings, dtype=torch.float32)
    
    return encodings_tensor

    