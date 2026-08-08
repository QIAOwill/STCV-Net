from torch import nn

# 定义均方误差损失函数(MSE Loss)
def MSELoss(predict,target): return (predict-target).pow(2).mean()

# 定义平均绝对误差损失函数(MAE Loss)
def MAELoss(predict,target): return (predict-target).abs().mean()
    
# 有选择的计算均方误差损失(MSE Loss)
def MSELoss_mask(predict,target,mask): return (predict-target)[mask].pow(2).mean()
# 有选择的计算平均绝对误差损失(MAE Loss)
def MAELoss_mask(predict,target,mask): return (predict-target)[mask].abs().mean()
# 有选择的计算平均绝对百分比误差损失(MAPE Loss) 
def MAPELoss_mask(predict,target,mask): return 100*((predict-target)[mask]/target[mask]).abs().mean()

# 定义一个用于计算各种评估指标的类
class Metrics(nn.Module):
    def __init__(self, max_min):
        """
        初始化Metrics类, 保存max_min用于放大结果中的某些指标.
        
        Args:
            max_min (float): 最大最小值
        """
        super().__init__()  # 调用父类的构造函数
        self.max = max_min[0]  # 保存max值
        self.min = max_min[1]  # 保存min值
        self.extreme = max_min[0] - max_min[1]
        # 由于数据中最小值为0, 因此采用极值就可得到原始数据的误差指标
        self.a=(self.extreme**2, self.extreme, self.extreme**2, self.extreme, 1)

    def forward(self, predict, target):
        """
        计算多种评估指标.
        
        Args:
            predict (tensor): 预测值张量
            target (tensor): 目标值张量
        
        返回:
        tuple: 包含多个评估指标的元组
        """
        # 设置筛选机制
        mask = target > 5/self.extreme
        # 用系数计算原始数据的误差指标
        return (self.a[0] * MSELoss(predict, target).item(),
                self.a[1] * MAELoss(predict, target).item(),
                self.a[2] * MSELoss_mask(predict,target,mask).item(),
                self.a[3] * MAELoss_mask(predict,target,mask).item(),
                self.a[4] * MAPELoss_mask(predict,target,mask).item(),
                )

# 定义一个用于格式化输出评估指标的函数
def Format_Metrics(metrics):
    """
    格式化输出评估指标结果为字符串.
    
    Args:
        metrics (tuple): 包含评估指标的元组 (MSE, MAE, MAPE)
    
    返回:
    str: 格式化后的评估指标字符串
    """
    (mse, mae, mse_mask, mae_mask, mape_mask) = metrics
    # 格式化为字符串, 保留三位小数, 便于输出显示
    return (f'MSE={mse:5.3f} | MAE={mae:5.3f}'+ 
            f' | MSE_Mask={mse_mask:5.3f} | MAE_Mask={mae_mask:5.3f} | MAPE_Mask={mape_mask:5.3f}%')
