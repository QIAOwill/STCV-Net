import copy
import gc
import os
import time

import torch

import Path
from Process import Data_Loader as DL
from Process import Learning, Loss, Paras
from Process.Time import output_duration


param_dict = dict(
    num_history=[14],             # 历史数据长度
    future_steps=[8],             # 未来预测步长
    batch_size=[8, 16, 32],       # 网格搜索批次大小
    split_bound=[0.8],            # 测试期之前的数据中训练集比例

    dim_time=[8],
    gcn_list=[(64, 256, 64)],
    dim_edge=[16],
    dim_lstm=[128],
    num_heads=[4],
    num_blocks=[4],
    map_layers=[(64, 256, 64)],
    pre_impact=[2],
    slope=[0.2],
    num_epochs=[2],
    num_workers=[0],

    device=['cuda' if torch.cuda.is_available() else 'cpu'],
    num_cities=[''],
    dim_time_in=[''],
    dim_edge_in=[''],
    save_point=[40],              # 保留旧参数以兼容 Result；最佳验证 checkpoint 现在总会保存
    seed=[54321],
    start_time=[time.time()],
    end_time=[''],
)

Model_part = ['HFC', 'Main_Model'][1]


def _make_scaler(device):
    enabled = str(device).startswith('cuda')
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _prepare_dataset(para, data_cache):
    """利用缓存准备当前参数组合的数据，不重复读取磁盘文件。"""
    train_ratio = float(para.split_bound)
    Global, Edge, Flow, HFC_result, max_min = DL.Read_file(
        para.pre_impact,
        para.slope,
        train_ratio=train_ratio,
        future_steps=para.future_steps,
        cache=data_cache,
    )

    para.num_cities = Flow.shape[-1]
    test_start = Flow.shape[0] - para.future_steps
    train_end = int(test_start * train_ratio)
    data_split_boundaries = (train_end, test_start)
    para.data_split_boundaries = data_split_boundaries

    dataset, dim_time_in, dim_edge_in = DL.Split_data(
        Global, Edge, Flow, HFC_result,
        para.num_history,
        para.future_steps,
        data_split_boundaries,
    )
    para.dim_time_in = dim_time_in
    para.dim_edge_in = dim_edge_in
    return dataset, max_min


def _write_grid_summary(summaries, best_summary):
    os.makedirs(Path.result_path, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(Path.result_path, f'Grid Search Summary_{stamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('STCV-Net hyperparameter search summary\n')
        f.write('Only validation metrics are used to choose the best run; test set is evaluated once afterwards.\n')
        f.write('=' * 120 + '\n')
        for idx, item in enumerate(summaries, start=1):
            marker = '  <-- BEST' if item['run_path'] == best_summary['run_path'] else ''
            f.write(
                f"{idx:03d} | {os.path.basename(item['run_path'])} | "
                f"best_epoch={item['best_epoch']} | {item['selection_metric']}={item['selection_score']:.8f}"
                f"{marker}\n"
            )
            params = item['parameters']
            f.write(
                f"      batch_size={params.get('batch_size')}, num_history={params.get('num_history')}, "
                f"future_steps={params.get('future_steps')}, pre_impact={params.get('pre_impact')}, "
                f"slope={params.get('slope')}\n"
            )
    return summary_path


if __name__ == '__main__':
    if Model_part == 'HFC':
        from Model.HFC import HFC_Model

        namelist = ['Prophet']
        precict_day = 8
        citys_range = None
        HFC_Model(
            namelist[0],
            precict_day=precict_day,
            citys_range=citys_range,
            data_path=os.path.join(Path.root_file, 'data.txt'),
            predict_path=os.path.join(Path.root_file, 'predict.txt'),
            od_path=os.path.join(Path.root_file, 'delta OD.txt'),
        )

    if Model_part == 'Main_Model':
        from Model import Main_Model

        program_start = time.time()
        parameters_list = Paras.list_of_param_dicts(param_dict)

        # 关键优化：整个网格搜索共用同一个缓存。
        # 第一组参数读取一次 Data 文件，后续只要数据预处理参数未改变就直接复用 Tensor。
        data_cache = DL.DataCache()
        run_summaries = []
        best_summary = None
        best_params = None

        for run_index, parameters in enumerate(parameters_list, start=1):
            print('\n' + '#' * 100)
            print(f'参数组合 {run_index}/{len(parameters_list)}')
            print('#' * 100)

            para = Paras.Args(copy.deepcopy(parameters))
            para.start_time = time.time()
            dataset, max_min = _prepare_dataset(para, data_cache)

            # 在模型初始化前重置随机种子，保证不同超参数组合的初始化可复现。
            Learning.set_seed(para.seed)
            model = Main_Model.Main_Model(para)
            optimizer = torch.optim.Adam(model.parameters(), 3e-4)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            scaler = _make_scaler(para.device)

            summary = Learning.Loop(
                model,
                para,
                optimizer,
                scheduler,
                scaler,
                dataset,
                torch.nn.L1Loss(),
                Loss.Metrics(max_min),
                batch_size=para.batch_size,
                num_epochs=para.num_epochs,
                num_workers=para.num_workers,
                device=para.device,
                max_min=max_min,
            )
            run_summaries.append(summary)

            if best_summary is None or summary['selection_score'] < best_summary['selection_score']:
                best_summary = summary
                best_params = copy.deepcopy(para.__dict__)

            # 每个候选模型训练完成后释放 GPU；最佳模型通过 Run_N/BestModel.pt 保留，不占显存。
            del model, optimizer, scheduler, scaler, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_summary is None:
            raise RuntimeError('没有成功完成任何超参数组合。')

        print('\n' + '=' * 100)
        print('根据验证集选出的最佳参数组合:')
        print(f"Run目录: {best_summary['run_path']}")
        print(f"选择指标: {best_summary['selection_metric']} = {best_summary['selection_score']:.8f}")
        for k, v in best_params.items():
            if k not in ('start_time', 'end_time', 'output_path'):
                print(f'{k}: {v}')
        print('=' * 100)

        # 测试集只在网格搜索完成后，对最佳验证组合评估一次；结果写回该组合原来的 Run_N。
        best_para = Paras.Args(copy.deepcopy(best_params))
        best_dataset, best_max_min = _prepare_dataset(best_para, data_cache)
        best_model = Main_Model.Main_Model(best_para)
        test_dates = data_cache.test_dates(best_para.future_steps)

        Learning.Test_and_Save(
            best_model,
            best_para,
            best_dataset,
            Loss.Metrics(best_max_min),
            best_summary,
            best_max_min,
            city_list=data_cache.city_list,
            test_dates=test_dates,
            batch_size=best_para.batch_size,
            num_workers=best_para.num_workers,
            device=best_para.device,
        )

        grid_summary_path = _write_grid_summary(run_summaries, best_summary)
        print(f'网格搜索摘要已保存: {grid_summary_path}')

        del best_model, best_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_duration(program_start, time.time(), print_time=True)
