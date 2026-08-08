import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import ModuleList

class Virtual_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_list, output_dim):
        """
        初始化 GCN 模型
        
        Args:
            input_dim (int): 输入节点特征的维度
            hidden_list (int): 隐藏层维度列表
            output_dim (int): 输出节点特征的维度
        """
        super(Virtual_GCN, self).__init__()
        
        # 添加初始图卷积
        self.conv_list = ModuleList([GCNConv(input_dim, hidden_list[0])])
        
        # 循环添加后续图卷积
        for i in range(len(hidden_list) - 1):
            self.conv_list.append(GCNConv(hidden_list[i], hidden_list[i + 1]))
        
        # 添加最终图卷积
        self.conv_list.append(GCNConv(hidden_list[-1], output_dim))

    def cosine_similarity_matrix(self, Flow):
        """
        根据时间序列计算相似度矩阵
        
        Args:
            Flow (torch.tensor): 流量数据
        
        Returns:
            similarity_matrix (torch.tensor): 相似度矩阵
        """
        # 获取数据维度
        time, _, _ = Flow.shape
        x = Flow.view(time, -1)  # (time, N**2)

        # 计算余弦相似度
        norm_x = F.normalize(x, dim=1)  # 在时间维度上归一化
        # 对归一化数据进行乘积，得到余弦相似度
        similarity_matrix = torch.einsum('ti,tj->ij', norm_x, norm_x)
        return similarity_matrix

    def gcn_forward(self, node_features, edge_index, edge_weight):
        """
        图卷积层前向传播
        
        Args:
            node_features (torch.tensor): 节点特征
            edge_index (torch.tensor): 边的对儿
            edge_weight (torch.tensor): 边的权重
        
        Returns:
            x: GCN 输出的节点特征
        """
        # 根据卷积列表对数据进行卷积操作
        for conv in self.conv_list:
            node_features = conv(node_features, edge_index, edge_weight)
            node_features = torch.relu(node_features)
        return node_features

    def forward(self, Flow, Edge):
        """
        前向传播函数
        
        Args:
            Flow: 流量数据, 大小为 (batch_size, Time, city, city)
            Edge: 边特征(将OD视为节点时, 表示节点特征), 大小为 (batch_size, city, city, embedding)
        
        Returns:
            outputs: 每个图的节点特征输出, 大小为 (batch_size, city**2, output_dim)
        """
        # 获取特征数据维度
        batch_size, city, city, embedding = Edge.shape
        result_list = []  # 初始化结果列表

        for batch in range(batch_size):
            # 计算邻接相似度矩阵, 其大小为 (batch_size, city ** 2, city ** 2)
            adj_matrix = self.cosine_similarity_matrix(Flow[batch])
            
            # 获得节点特征数据, 其大小为 (batch_size, city**2, embedding)
            node_features = Edge[batch].view(city * city, embedding)
            
            # 构造边索引 (edge_index)
            edge_index = adj_matrix.nonzero(as_tuple=False).T  # 获取非零的相似度对（边）

            # 构造边权重 (edge_weight)，即相似度值
            edge_weight = adj_matrix[edge_index[0], edge_index[1]]

            # 图卷积前向传播
            x = self.gcn_forward(node_features, edge_index, edge_weight)
            
            # 添加当前结果
            result_list.append(x.view(1, city, city, -1))
        
        return torch.cat(result_list)

class OD_Similarity(torch.nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(OD_Similarity, self).__init__()

    def OD_Cos_Matrix(self, batch_flow):
        """
        根据时间序列计算相似度矩阵
        
        Args:
            batch_flow (torch.tensor): 一个批次的流量数据

        Returns:
            similarity_matrix (torch.tensor): 计算得到的相似度矩阵
        """
       
        # 获取OD数据（出发点 O 和到达点 D 的流量汇总）
        O_data = batch_flow.sum(dim=1)
        D_data = batch_flow.sum(dim=2)
        # 计算归一化的流量数据（单位化流量）
        O_norm = F.normalize(O_data, dim=1)  
        D_norm = F.normalize(D_data, dim=1)

        # 计算余弦相似度
        similarity_O = torch.einsum('ti,tj->ij', O_norm, O_norm)
        similarity_D = torch.einsum('ti,tj->ij', D_norm, D_norm)
        
        return torch.stack([similarity_O, similarity_D], dim = 0)  # 返回计算出的相似度矩阵

    def forward(self, Flow):
        """
        前向传播函数，计算整个批次数据的相似度矩阵
        
        Args:
            Flow (torch.tensor): 输入的流量数据，形状为 (B, N, T, D)，
                                 B 是批次大小，N 是样本数，T 是时间步数，D 是特征数（维度）
        Returns:
            torch.tensor: 计算得到的相似度矩阵，形状为 (B, 1, N, N)，
                          B 是批次大小，N 是样本数
        """
        result_list = []  # 初始化结果列表

        # 遍历批次中的每个样本
        for batch in range(Flow.shape[0]):  # Flow.shape[0] 是批次大小
            # 对当前批次数据计算相似度矩阵
            OD_sim = self.OD_Cos_Matrix(Flow[batch])
            result_list.append(OD_sim.unsqueeze(0))  # 将每个批次的相似度矩阵添加到结果列表中
        
        # cat 会把每个批次的相似度矩阵沿着批次维度拼接
        return torch.cat(result_list, dim = 0)

