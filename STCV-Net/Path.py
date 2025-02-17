import os

# 数据文件根目录地址
root_file = r'../Data/'
# 邻接矩阵数据地址
adjacent_file = root_file +'delta adjacent.txt'
# 联系强度与相似性数据地址
connection_file = root_file +'delta connection.txt'
# 客流量数据地址
flow_file = root_file +'delta OD.txt'
# 城市属性结构数据地址
structure_file = root_file +'delta_structure.txt'
# HFC模型的运行结果
HFC_result_file = root_file +'predict.txt'

# 结果保存地址
result_path =r'../Run_File/'

# 文件输出地址更新
def update_path(result_path):
    """
    更新输出文件路径
    """
    # 文件名从 Run_1 开始
    i = 1
    new_path = r'%sRun_%s/'%(result_path, i)
    # 判断 Run_1 文件是否存在，不存在则新建
    if os.path.exists(new_path) == False:
        os.makedirs(new_path)
    else:
        while (os.path.exists(new_path)):
            # 获得文件中的文件数目
            files_num = os.listdir(new_path)
            # 文件夹为空，则不需要新建文件夹
            if len(files_num) == 0:
                break
            i += 1
            new_path = r'%sRun_%s/'%(result_path, i)
        # 如果路径没有文件夹，则新建文件夹
        if os.path.exists(new_path) == False:
            os.makedirs(new_path)

    return new_path