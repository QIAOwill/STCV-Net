import torch
from torch import nn
from Model.Virtual_GCN import Virtual_GCN, OD_Similarity
from torch.nn.functional import leaky_relu,relu

a = 0.2  # Leaky ReLU激活函数的负斜率参数

# 定义一个Prompter类, 继承自nn.Module (神经网络的基础模块) 
class Prompter(nn.Module):
    def __init__(self, dim_left, dim_right, dim_time, fit_E):
        """
        初始化Prompter类的构造函数.
        
        Args:
            dim_left (int): 左侧维度
            dim_right (int): 右侧维度
            fit_E (bool): 一个布尔变量, 决定是否在最后返回的P张量中添加一个维度
        """
        super().__init__()  # 调用父类的构造函数
        self.fit_E = fit_E  # 赋值fit_E参数
        # 初始化S矩阵, 随机生成一个大小为(dim_left, dim_right)的参数, 乘以1e-3缩小初始值
        self.S = nn.Parameter(1e-3 * torch.randn(dim_left, dim_right))
        # 初始化L1、L2、R1、R2四个参数矩阵, 使用Kaiming正态初始化
        self.L1 = nn.Parameter(nn.init.kaiming_normal_(
                    torch.empty(8 * dim_time, dim_time)))
        
        self.L2 = nn.Parameter(nn.init.kaiming_normal_(
                    torch.empty(dim_left, 8 * dim_time)))
        
        self.R1 = nn.Parameter(nn.init.kaiming_normal_(
                    torch.empty(dim_time, 8 * dim_time)))
        
        self.R2 = nn.Parameter(nn.init.kaiming_normal_(
                    torch.empty(8 * dim_time, dim_right)))

    def forward(self, G):
        """
        前向传播函数.

        Args:
            G (torch.tensor): 输入时间特征张量

        Returns:
            tuple: 返回 G 和 E 张量
        """
        # 通过矩阵相乘和Leaky ReLU激活函数计算P
        P = self.S + leaky_relu(self.L2 @ leaky_relu(self.L1 @ G @ self.R1, a) @ self.R2, a)
        # 如果fit_E为True, 返回P张量并在倒数第三个维度上扩展一维; 否则直接返回P
        return P.unsqueeze(-3) if self.fit_E else P


# 定义Prompters类, 继承自nn.Module
class Prompters(nn.Module):
    def __init__(self, dim_left, dim_right, dim_time, 
                 direction, fit_E):
        """
        初始化Prompters类的构造函数.
        
        Args:
            dim_left (int): 左侧维度
            dim_right (int): 右侧维度
            dim_time (int): 时间编码维度, 
            direction (str): 方向, 值可以为'L' (左) 或'R' (右) 
            fit_E (bool): 一个布尔变量, 决定是否在最后返回的P张量中添加一个维度
        """
        super().__init__()  # 调用父类的构造函数
        self.direction = direction  # 赋值方向参数
        # n是一个基于dim_left和dim_right的值, 确保n为16的倍数且在合理范围内
        n = max(16, min(dim_left, dim_right) // 16 * 16)
        # 根据方向创建Prompter的ModuleList
        if direction == 'R':
            self.prompter_list = nn.ModuleList(
                    [Prompter(dim_left, n, dim_time, fit_E), 
                     Prompter(n, dim_right, dim_time, fit_E)]
                )
        if direction == 'L':
            self.prompter_list = nn.ModuleList(
                    [Prompter(n, dim_right, dim_time, fit_E), 
                     Prompter(dim_left, n, dim_time, fit_E)]
                )

    def forward(self, G):
        """
        前向传播函数, 计算输出.

        Args:
            G (torch.tensor): 输入时间特征张量
        
        Return: 
            返回方向和Prompter的输出列表
        """
        # 遍历Prompters列表, 计算每个Prompter的输出, 并返回方向和计算结果
        return (self.direction, [Prompter(G) for Prompter in self.prompter_list])

def Injector(input_tensor : torch.tensor, prompters):
    """
    Injector 函数, 用于根据传入的 promperters (Prompter 模型列表或特定字符串) 对输入张量 input_tensor 进行映射操作. 
    
    Args:
        input_tensor (torch.Tensor): 输入的张量
        prompters (list): 包含 Prompter 模型或方向字符串的列表
    
    Returns:
        torch.Tensor: 映射操作后的张量
    """
    # 如果 prompters 的第一个元素是字符串类型
    if isinstance(prompters[0], str):
        # 如果方向为 'R', 则执行右侧的映射操作, 先通过第一个 Prompter, 再通过第二个 Prompter
        if prompters[0] == 'R': 
            return leaky_relu(leaky_relu(input_tensor @ prompters[1][0], a) @ prompters[1][1], a)
        # 如果方向为 'L', 则执行左侧的映射操作, 先通过第二个 Prompter, 再通过第一个 Prompter
        if prompters[0] == 'L': 
            return leaky_relu(prompters[1][1] @ leaky_relu(prompters[1][0] @ input_tensor, a), a)
        # 如果方向为 'R_long', 则遍历 promperters 列表, 依次通过每个 Prompter 执行右侧映射操作
        if prompters[0] == 'R_long':
            for i in range(len(prompters[1])): 
                input_tensor = relu(input_tensor @ prompters[1][i], a)
            return input_tensor
        # 如果方向为 'L_long', 则遍历 prompters 列表, 依次通过每个 Prompter 执行左侧映射操作
        if prompters[0] == 'L_long':
            for i in range(len(prompters[1])): 
                input_tensor = leaky_relu(prompters[1][i] @ input_tensor, a)
            return input_tensor
    
    # 如果 promperters 的第一个元素是嵌套列表
    else:
        # 当第一个 Prompter 的方向为 'L' 且第二个为 'R', 执行左右组合的映射操作
        if prompters[0][0] == 'L' and prompters[1][0] == 'R':
            return leaky_relu(prompters[0][1][1] @ leaky_relu(
                prompters[0][1][0] @ input_tensor @ prompters[1][1][0], a) @ prompters[1][1][1], a)
        
        # 当第一个 Prompter 的方向为 'R' 且第二个为 'L', 执行左右组合的映射操作
        if prompters[1][0] == 'L' and prompters[0][0] == 'R':
            return leaky_relu(prompters[1][1][1] @ leaky_relu(
                prompters[1][1][0] @ input_tensor @ prompters[0][1][0], a) @ prompters[0][1][1], a)

# 定义 Projector 类, 继承自 nn.Module
class Projector(nn.Module):
    def __init__(self, para):
        """
        初始化 Projector 类

        Args:
            dim_time_in (int): 输入的时间embedding特征维度
            dim_edge_in (int): 输入的属性特征维度
            init (float): 初始化的缩放因子, 默认为 0.5 / 全局维度
        """
        super().__init__()
        self.init = 0.6/ para.dim_time  # 初始化因子
        self.OD_Similarity = OD_Similarity()

        # 定义时间embedding特征到 G 的变换矩阵, 使用 Kaiming 正态初始化
        self.Global_to_G = nn.ParameterList([
            nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(para.dim_time_in, 4 * para.dim_time))),

            nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(4 * para.dim_time, 8 * para.dim_time))),

            nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(8 * para.dim_time, 4 * para.dim_time))),

            nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(4 * para.dim_time, para.dim_time)))
        ])

        # 定义属性特征的投影 (Prompter 模型) , 用于处理属性特征的预投影
        self.pre_project_Edge = Prompters(para.dim_edge_in, para.dim_edge // 2, 
                                          para.dim_time, 'R', True)
        # 定义历史OD流量数据的投影
        self.pre_projectleaky_reluLOW = Prompters(para.num_history + 2, para.dim_edge // 2, # 这里的num_history+1是有1个OD相似数据
                                              para.dim_time, 'R', True)

    def forward(self, Global, Edge, Flow):
        """
        前向传播函数, 生成 G 和 E 张量

        Args:
            Global (torch.Tensor): 时间embedding特征输入
            Edge (torch.Tensor): 属性特征输入
            Flow (torch.Tensor): OD流量数据输入

        Returns:
            tuple: 返回 G 和 E 张量
        """
        multi_Flow = torch.cat((self.OD_Similarity(Flow), Flow), dim = 1)
        # 将时间embedding特征进行变换, 使用 Prompter 进行多步映射
        G = torch.stack([x.diag() for x in Injector(self.init * Global, ['R_long', self.Global_to_G])])
        # 将属性特征和OD流量数据进行映射, 并拼接在一起
        E = torch.concatenate((Injector(Edge, self.pre_project_Edge(G)),
                               Injector(multi_Flow.permute(0, 2, 3, 1), self.pre_projectleaky_reluLOW(G))), -1)
        return (G, E)


# 定义 Update_O 类, 继承自 nn.Module
class Update_O(nn.Module):
    def __init__(self, para):
        """
        初始化 Update_O 类, 用于更新属性特征
        """
        super().__init__()
        # 定义两个 Prompters 模型, 分别处理区域和边缘维度的变换
        self.W_L = Prompters(para.num_cities, para.num_cities, para.dim_time, 'L', True)
        self.W_R = Prompters(para.dim_edge, para.dim_edge, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 更新属性特征

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 更新后的属性特征
        """
        # 使用 Injector 函数将时间embedding特征和属性特征映射并返回更新结果
        return Injector(E.transpose(1, 2), (self.W_L(G), self.W_R(G))).transpose(1, 2)


# 定义 Update_D 类, 继承自 nn.Module
class Update_D(nn.Module):
    def __init__(self, para):
        """
        初始化 Update_D 类, 用于更新OD流量数据
        """
        super().__init__()
        # 定义两个 Prompters 模型, 分别处理区域和边缘维度的变换
        self.W_L = Prompters(para.num_cities, para.num_cities, para.dim_time, 'L', True)
        self.W_R = Prompters(para.dim_edge, para.dim_edge, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 更新OD流量数据

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 更新后的OD流量数据
        """
        return Injector(E, (self.W_L(G), self.W_R(G)))


# 定义 Update 类, 继承自 nn.Module
class Update(nn.Module):
    def __init__(self, para):
        """
        初始化 Update 类, 整合边缘和流量的更新模块
        """
        super().__init__()
        # 定义 Update_O 和 Update_D 两个更新模块
        self.Update_O = Update_O(para)
        self.Update_D = Update_D(para)
        # 定义用于处理更新后的特征的 Prompter 模型
        self.W = Prompters(2 * para.dim_edge, para.dim_edge, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 更新边缘和OD流量数据

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 更新后的特征
        """
        # 先分别更新边缘和OD流量数据, 然后将结果拼接, 并通过 W 进行映射
        return Injector(torch.concatenate((self.Update_O(G, E), self.Update_D(G, E)), -1), self.W(G))

class Time_Varying_Attention(nn.Module):
    def __init__(self, para):
        super().__init__()
        # 设置时间长度、边特征维度和城市数量
        self.time_length = para.num_history  # 时间步数
        self.dim_edge = para.dim_edge        # 边的特征维度
        self.num_cities = para.num_cities      # 城市数量

        # 定义用于时间变化捕捉 (Time Varying Captor) 的LSTM模块
        self.TV_captor = nn.LSTM(self.num_cities**2, para.dim_lstm)  
        # 定义聚合模块，用于聚合时间维度上的特征
        self.Aggregation = Prompters(para.dim_lstm, 1, para.dim_time, 'R', False)

        # 定义特征维度向时间维度转换的映射模块
        self.edge_to_day = Prompters(self.dim_edge, self.time_length, para.dim_time, 'R', True)
        # 定义时间维度向特征维度转换的映射模块
        self.day_to_edge = Prompters(self.time_length, self.dim_edge, para.dim_time, 'R', True)

    def get_hidden_set(self, E_day):
        """
        获取LSTM的隐藏状态集合
        Args:
            E_day: 输入的每日边特征张量
        Returns:
            各时间步的隐藏状态集合
        """
        # 调整输入形状为 (batch_size, time_length, num_cities**2)，LSTM 才能处理序列数据
        # 包含每个时间步、每个样本的隐藏状态, 形状为 (batch_size, time_length, hidden_dim)
        return self.TV_captor(E_day.view(-1, self.time_length, self.num_cities**2))[0] 

    def forward(self, G, E):
        """
        前向传播函数, 计算属性特征的注意力机制

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性特征

        Returns:
            torch.Tensor: 应用注意力机制后的属性特征
        """
        # 将特征边信息 E 映射到时间维度上
        E_day = Injector(E, self.edge_to_day(G))
        # 使用LSTM获取时间变化特征的隐藏状态，随后聚合
        A = Injector(self.get_hidden_set(E_day), self.Aggregation(G))
        # 将聚合后的结果与 E_day 相乘，并注入结果
        A = A.unsqueeze(-1).permute(0,2,3,1).repeat(1, E_day.shape[1], E_day.shape[2], 1)
        return Injector(A * E_day, self.day_to_edge(G))

# 定义 Attention_O 类, 继承自 nn.Module
class Attention_O(nn.Module):
    def __init__(self, para):
        """
        初始化 Attention_O 类, 用于计算属性特征的注意力权重
        """
        super().__init__()
        # 定义用于聚合属性特征的 Prompter 模型
        self.Aggregation = Prompters(para.dim_edge, 1, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 计算属性特征的注意力机制

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性特征

        Returns:
            torch.Tensor: 应用注意力机制后的属性特征
        """
        A = Injector(E.transpose(1, 2), self.Aggregation(G)).squeeze(-1).softmax(-1).unsqueeze(-3)
        return (A @ E.transpose(1, 2)).transpose(1, 2)


# 定义 Attention_D 类, 继承自 nn.Module
class Attention_D(nn.Module):
    def __init__(self, para):
        """
        初始化 Attention_D 类, 用于计算OD流量数据的注意力权重
        """
        super().__init__()
        # 定义用于聚合OD流量数据的 Prompter 模型
        self.Aggregation = Prompters(para.dim_edge, 1, para.dim_time, 'R', True)


    def forward(self, G, E):
        """
        前向传播函数, 计算OD流量数据的注意力机制

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): OD流量数据

        Returns:
            torch.Tensor: 应用注意力机制后的OD流量数据
        """
        A = Injector(E, self.Aggregation(G)).squeeze(-1).softmax(-1).unsqueeze(-3)
        return A @ E


# 定义 Attention 类, 继承自 nn.Module
class Attention(nn.Module):
    def __init__(self, para):
        """
        初始化 Attention 类, 整合边缘和流量的注意力机制
        """
        super().__init__()
        # 定义边缘和流量的注意力模块
        self.Attention_Time = Time_Varying_Attention(para)
        self.Attention_O = Attention_O(para)
        self.Attention_D = Attention_D(para)
        # 定义用于整合注意力结果的 Prompter 模型
        self.W = Prompters(3 * para.dim_edge, para.dim_edge, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 整合边缘和OD流量数据的注意力机制

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 应用注意力机制后的特征
        """
        return Injector(torch.concatenate((self.Attention_O(G, E), 
                                           self.Attention_D(G, E),
                                           self.Attention_Time(G, E)),
                                             -1), self.W(G))

# 定义 MHA 类 (多头注意力) , 继承自 nn.Module
class MHA(nn.Module):
    def __init__(self, para):
        """
        初始化 MHA 类, 用于多头注意力机制

        """
        super().__init__()
        # 定义多个 Attention 模块, 每个代表一个头
        self.Attentions = nn.ModuleList([Attention(para) for _ in range(para.num_heads)])
        # 定义用于聚合多头注意力结果的 Prompter 模型
        self.Weights = Prompter(para.num_heads, 1, para.dim_time, True)

    def forward(self, G, E):
        """
        前向传播函数, 计算多头注意力机制

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 多头注意力结果
        """
        # 此处将多头注意力机制的得到的结果合并后, 再转化为原来的形状
        return (torch.stack(([Attention(G, E) for Attention in self.Attentions]), -1) @
                    self.Weights(G).softmax(-2).unsqueeze(-3)).squeeze(-1)

# 定义 FFN 类 (前馈神经网络) , 继承自 nn.Module
class FFN(nn.Module):
    def __init__(self, para):
        """
        初始化 FFN 类, 定义前馈神经网络模块
        """
        super().__init__()
        # 定义用于特征变换的 Prompter 模型
        self.W = Prompters(para.dim_edge, para.dim_edge, para.dim_time, 'R', True)

    def forward(self, G, E):
        """
        前向传播函数, 前馈神经网络的计算

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 通过前馈神经网络后的特征
        """
        return Injector(E, self.W(G))

# 定义 Transformer_Block 类, 继承自 nn.Module
class Transformer_Block(nn.Module):
    def __init__(self, para):
        """
        初始化 Transformer_Block 类, 定义单个 Transformer 块
        """
        super().__init__()
        # 定义更新模块、多头注意力机制和前馈神经网络
        self.Update = Update(para)
        self.MHA = MHA(para)
        self.FFN = FFN(para)

    def forward(self, G, E):
        """
        前向传播函数, 执行 Transformer 块的计算

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 通过 Transformer 块计算后的特征
        """
        E = E + self.Update(G, E)  # 先进行特征更新
        E = E + self.MHA(G, E)  # 然后应用多头注意力机制
        E = E + self.FFN(G, E)  # 最后通过前馈神经网络
        return E

# 定义 Transformer 类, 继承自 nn.Module
class Transformer(nn.Module):
    def __init__(self, para):
        """
        初始化 Transformer 类, 定义多个 Transformer 块
        Args:
            para: 参数类;
        """
        super().__init__()
        # 定义多个 Transformer_Block 模块
        self.Transformer_Blocks = nn.ModuleList([
            Transformer_Block(para) for _ in range(para.num_blocks)
            ])

    def forward(self, G, E):
        """
        前向传播函数, 依次通过多个 Transformer 块

        Args:
            G (torch.Tensor): 时间embedding特征
            E (torch.Tensor): 属性和OD特征

        Returns:
            torch.Tensor: 通过多个 Transformer 块后的特征
        """
        for Transformer_Block in self.Transformer_Blocks:
            E = Transformer_Block(G, E)
        return E

# 定义 Main_Model 类, 继承自 nn.Module
class Main_Model(nn.Module):
    def __init__(self, para):
        """
        初始化 Main_Model 类, 定义整个模型结构

        Args:
            para: 参数类;
        """
        super().__init__()
        # 定义虚拟图卷积
        self.VGCN = Virtual_GCN(para.dim_edge, para.gcn_list, para.dim_edge)
        # 定义投影模块和 Transformer 模块
        self.Projector = Projector(para)
        self.Transformer = Transformer(para)
        # 定义用于最后处理的 Prompter 模型
        self.W = Prompters(para.dim_edge, para.future_steps, para.dim_time, 'R', True)
        # 获得结果的映射函数
        self.get_result_map(para)
    
    def get_result_map(self, para):
        """
        该函数用于生成一个映射列表，包含多个参数矩阵，这些矩阵用于将输入特征逐层映射到输出特征。
        每个矩阵通过 Kaiming 正态初始化以确保模型的稳定训练。

        Args:
            para: 参数对象.
        """
        map_list = []  # 用于存储每一层的映射矩阵
        input_dim = para.future_steps * 2  # 初始化输入维度

        # 遍历映射层配置，创建每一层的参数矩阵
        for map_i in para.map_layers:
            map_list.append(
                nn.Parameter(nn.init.kaiming_normal_(
                    torch.empty(input_dim, map_i))))  # 使用 Kaiming 正态初始化参数矩阵
            input_dim = map_i  # 更新输入维度为当前映射层的输出维度

        # 添加最后一层的参数矩阵，将最后一层的输出维度映射到指定维度
        map_list.append(
            nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(map_i, para.future_steps))))

        # 将映射列表转换为可训练的参数列表
        self.result_map = nn.ParameterList(map_list)

    def forward(self, input_data, HFC):
        """
        前向传播函数, 计算整个模型的输出

        Args:
            input_data (tuple): 包含时间embedding特征和属性特征的输入
            HFC_result (torch.tensor): HFC模型的结果;

        Returns:
            torch.Tensor: 最终的模型输出
        """
        Global, Edge, Flow = input_data
        # 首先通过 Projector 进行特征变换
        (G, E) = self.Projector(Global, Edge, Flow) 
        # 用虚拟图进行更新边的embedding
        E = self.VGCN(Flow, E)
        
        # 得到 Transformer模型的结果
        result1 = Injector(self.Transformer(G, E), self.W(G))
        # 获得 HFC 模型的结果
        result2 = HFC.permute(0,2,3,1)
        # 输出映射结果
        return Injector(torch.concatenate((result1, result2), -1), ['R_long', self.result_map])