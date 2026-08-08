import os

# 以当前代码文件位置为基准，避免“从不同工作目录点击 Main.py”导致相对路径跑偏。
MODEL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../STCV-Net
PROJECT_DIR = os.path.dirname(MODEL_CODE_DIR)                       # .../STCV-Net 2026

# 数据文件目录：默认与 STCV-Net 代码目录同级，即 .../STCV-Net 2026/Data/
root_file = os.path.join(PROJECT_DIR, 'Data')

adjacent_file = os.path.join(root_file, 'delta adjacent.txt')
connection_file = os.path.join(root_file, 'delta connection.txt')
flow_file = os.path.join(root_file, 'delta OD.txt')
structure_file = os.path.join(root_file, 'delta_structure.txt')
HFC_result_file = os.path.join(root_file, 'predict.txt')

# 每组超参数对应 Run_1 / Run_2 / ...
result_path = os.path.join(PROJECT_DIR, 'Run_File')


def update_path(base_result_path=None):
    """创建并返回下一个空的 Run_N 目录。"""
    base_result_path = result_path if base_result_path is None else base_result_path
    os.makedirs(base_result_path, exist_ok=True)

    i = 1
    while True:
        new_path = os.path.join(base_result_path, f'Run_{i}')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        if len(os.listdir(new_path)) == 0:
            return new_path
        i += 1
