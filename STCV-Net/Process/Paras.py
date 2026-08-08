from itertools import product as prod

"""
将包含列表的字典转换为所有可能组合的字典列表的函数。
例如：
    list_of_param_dicts({'a': [1, 2], 'b': [3, 4]}) ---> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
"""
def list_of_param_dicts(param_dict):
    """
    参数:
        param_dict (dict): 参数的字典
    """
    # 获取字典中每个键对应的所有值的笛卡尔积
    vals = list(prod(*[v for k, v in param_dict.items()]))
    # 获取每个键按照它的值的长度重复的键的组合
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    # 使用键和值的组合创建新的字典，并返回包含所有组合的字典列表
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]

"""
用于适配模型参数的Arguments类
"""
class Args():
    """
    参数:
        arg_dict (dict): 模型参数的字典
    """
    def __init__(self, arg_dict):
        # 将字典的键值对作为对象的属性
        self.__dict__ = arg_dict
        
    def __getitem__(self, key):
        # 使用 getattr 函数根据 key 返回类中对应的属性值
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"参数字典中不存在名为'{key}'的参数! ")
        
    # 定义 keys 方法，返回所有属性名
    def keys(self):
        return [attr for attr in self.__dict__.keys()]

    # 定义 __iter__ 方法，使得类可以迭代属性名
    def __iter__(self):
        return iter(self.keys())
