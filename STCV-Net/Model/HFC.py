import pandas as pd
import numpy as np
np.random.seed(54321)
from prophet import Prophet
from IPython.display import clear_output
from datetime import datetime, timedelta

# Prophet参数设置
def get_holidays(front_impact = 1, rear_impact = 1):
    """
    获得节假日信息
    front_impact: 假期的前影响
    rear_impact: 假期的后影响
    """
    # 节假日设置
    holidays = [
        {
            'name': "New Year's Day", 
            'month': 1, 
            'day': 1, 
            'lower_window': 1 + front_impact, 
            'upper_window': 1 + rear_impact, 
            'spcial_work_day': []
            },
        {
            'name': "Spring Festival", 
            'month': 1, 
            'day': 22, 
            'lower_window': 1 + front_impact, 
            'upper_window': 5 + rear_impact, 
            'spcial_work_day': ['2023-01-28', '2023-01-29',]
            },
        #{
        #    'name': "Tomb-Sweeping Day", 
        #    'month': 4, 
        #    'day': 5, 
        #    'lower_window': 0 + front_impact, 
        #    'upper_window': 0 + rear_impact, 
        #    'spcial_work_day': []
        #    },
        {
            'name': "Labour Day", 
            'month': 5, 
            'day': 1, 
            'lower_window': 5 + front_impact, 
            'upper_window': 7 + rear_impact, 
            'spcial_work_day': ['2023-04-23', '2023-05-06',]
            },
        {
            'name': "Dragon Boat Festival", 
            'month': 6, 
            'day': 22, 
            'lower_window': 0 + front_impact, 
            'upper_window': 2 + rear_impact, 
            'spcial_work_day': ['2023-06-25']
            },
        #{
        #    'name': "Mid-Autumn Festival", 
        #    'month': 9, 
        #    'day': 29, 
        #    'lower_window': 0 + front_impact, 
        #    'upper_window': 0 + rear_impact, 
        #    'spcial_work_day': []
        #    },
        {
            'name': "National Day", 
            'month': 10, 
            'day': 1, 
            'lower_window': 5 + front_impact, 
            'upper_window': 7 + rear_impact, 
            'spcial_work_day': []#['2023-10-07', '2023-10-08']
            },
        ]
    return holidays

def get_preDate(date, day):
    """
    计算给定日期的前 day 天

    date (str): 当前日期(yyyy-mm-dd);
    day (int): 往前追溯的天数;

    return (list): 包含前 day 天的字符串;
    """
    previous_set = [] # 存放前 day 天的日期
    for i in range(day, 0, -1):
        # 计算当前日期前第 i 天的日期
        previous_date = date - timedelta(days=i)
        # 将日期对象转换为字符串,并添加到列表中
        previous_set.append(previous_date.strftime('%Y-%m-%d'))
    
    return previous_set

def get_afterDate(date, day):
    """
    计算给定日期的后 day 天

    date (str): 当前日期(yyyy-mm-dd);
    day (int): 往后追溯的天数;

    return (list): 包含后 day 天的字符串;
    """
    after_set = [] # 存放后 day 天的日期
    for i in range(1, day + 1):
        # 计算当前日期后第 i 天的日期
        after_date = date + timedelta(days=i)
        # 将日期对象转换为字符串,并添加到列表中
        after_set.append(after_date.strftime('%Y-%m-%d'))
    
    return after_set

def get_datelist(holiday_dict, front_impact, rear_impact, year = 2023):
    """
    对每个节假日字典进行处理, 得到影响力日期List;

    holiday_dict (dict): 包含'month', 'day', 'lower_window', 'upper_window', 'spcial_work_day';
    front_impact (int): 假期前影响力;
    rear_impact (int): 假期后影响天数;
    year (int): 年份, 默认为2023年

    return : total_date(list): 所有日期字符串(str)的列表;
              date_order(list): 每天所属日期的第几天(int)列表;

    """
    date = '%s-%s-%s'%(year, holiday_dict['month'], holiday_dict['day'])
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date, '%Y-%m-%d')
    # 获取前后天数的 list
    previous_set = get_preDate(date, holiday_dict['lower_window'])
    after_set = get_afterDate(date, holiday_dict['upper_window'])

    # 计算该节假日的所有影响天数
    total_date = previous_set + [date.strftime('%Y-%m-%d')] + after_set

    # 计算假期的天数
    holiday_num = (len(total_date)-front_impact-rear_impact)
    
    # 将假期前的影响天数标记为 1, 假期后的影响天数标记为 2, 假期中的天数标记为"总天数 * 10 + 第几天"
    date_order = [1 for _ in range(front_impact)]
    date_order += [(holiday_num*10+i) for i in range(1,1+holiday_num)]
    date_order += [2 for _ in range(rear_impact)]

    # 加入因节假日而调休的周末日期,并将序号设为0
    total_date +=  holiday_dict['spcial_work_day']
    date_order += [0 for _ in range(len(holiday_dict['spcial_work_day']))]
    
    return total_date, date_order

def getHolidays_df(front_impact, rear_impact):
    """
    获取节假日的信息,包括日期和序号;

    front_impact (int): 每个节日前的影响天数;
    rear_impact (int): 每个节日后的影响天数;

    return : order (DataFrame): 一列“ds”, 一列日期序号;
    """
    # 获取假期字典
    holidays = get_holidays(front_impact = front_impact, rear_impact = rear_impact)
    holiday_dates = [] # 节假日日期列表
    date_order_set = [] # 日期序号列表
    
    # 对每个节日进行循环
    for day in holidays:
        total_date, date_order = get_datelist(day, front_impact, rear_impact)
        holiday_dates += total_date
        date_order_set += date_order

    # 得到 order_df
    order_df = pd.DataFrame({"ds": pd.to_datetime(holiday_dates),
                                "date_order": date_order_set})
    
    return order_df.reset_index(drop = True)

def make_columns(origin_df):
    """
    将每个事件列为一列

    order_df: 天数序号;

    return: 将每个事件列为一列的datafram;
    """
    holidays_df = origin_df.copy()
    # 获得所有标签
    value_list = holidays_df['date_order'].drop_duplicates().values.tolist()
    
    for order in value_list:
        if order == 0:
            # 标记调休的周六周天时间
            holidays_df['spcial_work_day'] = (holidays_df['date_order'] == order).astype(int)
            # 标记 “holiday” 事件, 即真正假期时间 + 前后影响时间
            holidays_df['holiday'] = (holidays_df['date_order'] != order).astype(int)
        else:
            # 标记前后假期时间和假期时间
            holidays_df['day_%s'%order] = (holidays_df['date_order'] == order).astype(int)
    
    del holidays_df['date_order']
    holidays_df = holidays_df.groupby('ds').sum().reset_index()
    
    return holidays_df

def Prophet_forecast(data_df: pd.DataFrame, 
                    start_date = '2023-01-01', 
                    precict_day = 7,
                    front_impact = 0, 
                    rear_impact = 0,):
    """
    根据输入的单变量时间序列, 采用Prophet模型进行预测
    
    参数: 
        data_df (DataFrame): 单变量时间序列数据;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;

    return: 预测模型, 预测结果 (tuple[NeuralProphet, DataFrame])
    """
    # 初始化 Prophet 模型
    model = Prophet(
                    growth='linear',
                    seasonality_mode='additive',
                    interval_width=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.9,
                    #holidays=holiday_set,
                )

    # 设置标准的列名
    column_df = data_df.copy()
    column_df = column_df.reset_index(drop = True)
    column_df.columns = ['y']
    column_df['ds'] = pd.date_range(start = start_date, periods = len(column_df))

    # 获取节假日数据
    holidays_df = make_columns(getHolidays_df(front_impact = front_impact, 
                                              rear_impact = rear_impact ))
    # 将预测数据与节假日数据结合
    data = pd.merge(column_df, holidays_df, how = 'left', on = 'ds')
    data.fillna(0, inplace=True)

    # 切分数据集
    train, test = data.iloc[:-precict_day,:], data.iloc[-precict_day:,:]
    #  添加额外的事项
    for col in holidays_df.columns:
        if col != 'ds':
            model.add_regressor(col)

    model.fit(train)

    # 测试集提取preheat特征
    predictions_test = model.predict(test.drop('y', axis=1))
    
    return predictions_test

def run_Prophet(od_df:pd.DataFrame, 
        citys_range = None, 
        start_date = '2023-01-01', 
        precict_day = 7, 
        front_impact = 0,
        rear_impact = 0,
        ):
    
    """
    对多变量的时间序列进行循环预测;

    参数: 
        od_df (DataFrame): OD序列数据;
        citys_range (int) : 执行预测的序列数范围;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;

    return: 原始数据, 预测结果 (tuple[DataFrame, DataFrame])
    
    """
    # 数据处理
    data_df = od_df.T.reset_index(drop=True) # 以城市对为列, 时间为行;
    # 设置列名
    data_df.columns = data_df.iloc[0] + ',' + data_df.iloc[1]
    data_df = data_df.drop([0, 1]).reset_index(drop=True)
    # 设置需要预测的城市对数量
    if citys_range == None:
        citys_range = [0, len(od_df)]
    # 筛选出需要的原始数据
    data_df = data_df.iloc[:,citys_range[0]:citys_range[1]]

    # 初始化预测结果
    precict_df = pd.DataFrame()

    # 对每个城市对进行预测
    citys = 0
    num = 1 # 预测进度
    while citys < (citys_range[1]-citys_range[0]):
        if num % 10 == 0:
            clear_output(wait=True)  # 清除之前的输出
        print('\n\n', '-'*100, '\n')
        print(' '*40, f"预测进度: {num}/{citys_range[1]-citys_range[0]}")
        print('\n', '-'*100, '\n\n')

        origin_df = data_df.iloc[:, citys:citys+1]
        # 调用模型进行预测
        forecast = Prophet_forecast(
                                    data_df = origin_df,
                                    start_date = start_date, 
                                    precict_day = precict_day,
                                    front_impact = front_impact, 
                                    rear_impact = rear_impact,
                                )
        # 将真实值和预测值加起来
        pre_data = forecast['yhat'].values.tolist()

        # 积累当前计算结果
        precict_df = pd.concat([precict_df, pd.DataFrame(pre_data),], axis = 1)
        citys += 1
        num += 1

    # 设置城市对的列名及时间
    precict_df.columns = data_df.columns

    return data_df, precict_df

def HFC_Model(run_name, precict_day = 7, citys_range = None, 
        data_path = 'data.txt',
        predict_path = 'predict.txt',
        od_path = 'delta OD.txt'):
    """
    对多变量的时间序列进行循环预测;

    参数:
        run_name (str): 函数名;
        precict_day (int): 预测天数, 默认为: 7;
        citys_range (int): 执行预测的序列数范围;

    return: 原始数据, 预测结果 (tuple[DataFrame, DataFrame])

    """
    od_df = pd.read_csv(od_path, sep = '\t', header = 0)
    # 算法运行
    if run_name == 'Prophet':
        data_df, precict_df = run_Prophet(od_df, 
                                        citys_range = citys_range, 
                                        start_date = '2023-01-01', 
                                        precict_day = precict_day, 
                                        front_impact = 0,
                                        rear_impact = 0,
                                )

    # 保存运行结果
    data_df.to_csv(data_path, sep = '\t', index=False)
    precict_df.to_csv(predict_path, sep = '\t', index=False)

if __name__ == '__main__':

    namelist = ['Prophet']
    precict_day = 8
    citys_range = None

    HFC_Model(
            namelist[0], 
            precict_day = precict_day, 
            citys_range = citys_range,
        )