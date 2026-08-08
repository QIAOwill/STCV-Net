import copy
import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import Path
from Process.Loss import Format_Metrics
from Process.Time import get_interval


def set_seed(seed=54321):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def denormalize_tensor(tensor, max_min):
    max_value, min_value = max_min
    return tensor * (max_value - min_value) + min_value


def _device_type(device):
    return str(device).split(':', 1)[0]


def Train(model, optimizer, scaler, dataloader, loss_fn, metrics_fn, device):
    model.train()
    n = len(dataloader)
    metrics = []
    use_amp = _device_type(device) == 'cuda'

    for data in dataloader:
        input_data, target, hfc = data
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=_device_type(device), dtype=torch.float16, enabled=use_amp):
            predict = model([x.to(device, non_blocking=True) for x in input_data],
                            hfc.to(device, non_blocking=True))
            target = target.permute(0, 2, 3, 1).to(device, non_blocking=True)
            loss = loss_fn(predict, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        metrics.append(metrics_fn(predict.detach(), target))

    return [sum(x) / max(1, n) for x in zip(*metrics)]


def Validate(model, dataloader, metrics_fn, device):
    model.eval()
    n = len(dataloader)
    metrics = []

    with torch.inference_mode():
        for data in dataloader:
            input_data, target, hfc = data
            predict = model([x.to(device, non_blocking=True) for x in input_data],
                            hfc.to(device, non_blocking=True))
            target = target.permute(0, 2, 3, 1).to(device, non_blocking=True)
            predict = torch.clamp_min(predict, 0)
            metrics.append(metrics_fn(predict, target))

    return [sum(x) / max(1, n) for x in zip(*metrics)]


def _make_dataloaders(dataset, batch_size, num_workers, device):
    pin_memory = _device_type(device) == 'cuda'
    return {
        'train': DataLoader(dataset['train'], batch_size=batch_size, pin_memory=pin_memory,
                            num_workers=num_workers, shuffle=True),
        'validate': DataLoader(dataset['validate'], batch_size=batch_size, pin_memory=pin_memory,
                               num_workers=num_workers, shuffle=False),
        'test': DataLoader(dataset['test'], batch_size=batch_size, pin_memory=pin_memory,
                           num_workers=num_workers, shuffle=False),
    }


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _save_checkpoint(model, path, epoch, validation_metrics):
    checkpoint_path = os.path.join(path, 'BestModel.pt')
    torch.save({
        'epoch': int(epoch),
        'model_state_dict': _cpu_state_dict(model),
        'validation_metrics': [float(x) for x in validation_metrics],
    }, checkpoint_path)
    return checkpoint_path


def _load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    return model.to(device), state


def _format_value(value):
    if isinstance(value, float):
        return f'{value:.10g}'
    return str(value)


def write_result_file(summary):
    """生成单次运行摘要。"""
    path = os.path.join(summary['run_path'], 'Result.txt')
    params = summary['parameters']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('-' * 100 + '\n')
        f.write(f"* 参数总量: {summary['n_params']}\n")
        f.write(f"run_path: {summary['run_path']}\n")
        f.write(f"start_time: {datetime.fromtimestamp(summary['start_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"end_time: {datetime.fromtimestamp(summary['end_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        h, m, s = get_interval(summary['end_time'] - summary['start_time'])
        f.write(f"time interval: {h}小时 {m}分钟 {s}秒\n")
        f.write('-' * 100 + '\n')
        f.write('Parameters:\n')
        for k, v in params.items():
            if k not in ('start_time', 'end_time'):
                f.write(f'{k}: {_format_value(v)}\n')

        f.write('-' * 100 + '\n')
        f.write('Hyperparameter selection uses VALIDATION set only.\n')
        f.write(f"best_epoch: {summary['best_epoch']}\n")
        f.write(f"validation_metrics: {Format_Metrics(summary['validation_metrics'])}\n")
        f.write(f"validation_mse: {summary['validation_metrics'][0]:.8f}\n")
        f.write(f"validation_rmse: {math.sqrt(max(summary['validation_metrics'][0], 0.0)):.8f}\n")
        f.write(f"validation_mae: {summary['validation_metrics'][1]:.8f}\n")
        f.write(f"validation_mse_mask: {summary['validation_metrics'][2]:.8f}\n")
        f.write(f"validation_mae_mask: {summary['validation_metrics'][3]:.8f}\n")
        f.write(f"validation_mape_mask: {summary['validation_metrics'][4]:.8f}\n")
        f.write(f"selection_metric: {summary['selection_metric']}\n")
        f.write(f"selection_score: {summary['selection_score']:.8f}\n")
        f.write(f"checkpoint: {summary['checkpoint_path']}\n")

        if summary.get('test_metrics') is not None:
            tm = summary['test_metrics']
            f.write('-' * 100 + '\n')
            f.write('Held-out TEST metrics (evaluated only for the best validation run):\n')
            for key in ('MSE', 'RMSE', 'MAE', 'MSE_Mask', 'RMSE_Mask', 'MAE_Mask', 'MAPE_Mask'):
                f.write(f'{key}: {tm[key]:.8f}\n')
    return path


def Loop(model, para, optimizer, scheduler, scaler, dataset, loss_fn, metrics_fn,
         batch_size=32, num_epochs=100, num_workers=0, device='cuda', max_min=(143350.0, 0.0)):
    """
    训练一个超参数组合。

    与旧版不同：
    - 每个组合总会创建 Run_N 并保存 BestModel.pt + Result.txt；
    - 只依据验证集选择最佳 epoch；
    - 此处绝不读取/评估测试结果，防止网格搜索阶段使用测试集。
    """
    del max_min  # 保留旧接口兼容；训练阶段不需要反归一化。
    set_seed(getattr(para, 'seed', 54321))
    run_start = time.time()
    run_path = Path.update_path(Path.result_path)
    para.output_path = run_path

    dataloader = _make_dataloaders(dataset, batch_size, num_workers, device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    best_score = float('inf')
    best_epoch = 0
    best_metrics = None
    checkpoint_path = os.path.join(run_path, 'BestModel.pt')

    print('=' * 100)
    print(f'本次参数输出目录: {run_path}')

    for epoch in range(1, 1 + num_epochs):
        print(f'训练轮次: {epoch:0>3d}/{num_epochs:0>3d}   学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        train_metrics = Train(model, optimizer, scaler, dataloader['train'], loss_fn, metrics_fn, device)
        validate_metrics = Validate(model, dataloader['validate'], metrics_fn, device)
        scheduler.step()

        val_mape = float(validate_metrics[4])
        # 正常情况下以验证集 MAPE_Mask 选最优；若 MAPE 因有效样本为空而 NaN，则用 RMSE 兜底。
        if np.isfinite(val_mape):
            current_score = val_mape
            current_metric = 'validation_MAPE_Mask'
        else:
            current_score = math.sqrt(max(float(validate_metrics[0]), 0.0))
            current_metric = 'validation_RMSE_fallback'

        if best_epoch == 0 or current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = copy.deepcopy(validate_metrics)
            selection_metric = current_metric
            checkpoint_path = _save_checkpoint(model, run_path, epoch, validate_metrics)

        print('-' * 150)
        print(f'  训练集指标:  {Format_Metrics(train_metrics)}')
        print(f'  验证集指标:  {Format_Metrics(validate_metrics)}')
        print('-' * 150, '\n')

    run_end = time.time()
    summary = {
        'run_path': run_path,
        'checkpoint_path': checkpoint_path,
        'best_epoch': best_epoch,
        'validation_metrics': [float(x) for x in best_metrics],
        'selection_metric': selection_metric,
        'selection_score': float(best_score),
        'n_params': int(n_params),
        'start_time': run_start,
        'end_time': run_end,
        'parameters': copy.deepcopy(para.__dict__),
        'test_metrics': None,
    }
    write_result_file(summary)

    print(f'最佳验证轮次: {best_epoch:0>3d}')
    print(f'最佳验证集指标: {Format_Metrics(best_metrics)}')
    print(f'已保存: {checkpoint_path}')
    print(f'已保存: {os.path.join(run_path, "Result.txt")}')
    return summary


def _collect_test_predictions(model, dataloader, device):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for input_data, target, hfc in dataloader:
            predict = model([x.to(device, non_blocking=True) for x in input_data],
                            hfc.to(device, non_blocking=True))
            predict = torch.clamp_min(predict, 0)
            target = target.permute(0, 2, 3, 1).to(device, non_blocking=True)
            predictions.append(predict)
            targets.append(target)
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def _raw_test_metrics(predict_raw, target_raw, normalized_metrics):
    err = predict_raw - target_raw
    abs_err = err.abs()
    mse = float((err ** 2).mean().item())
    mae = float(abs_err.mean().item())

    # 与 STCV-Net 原 Metrics 的含义一致：原始流量 > 5 时计算 masked 指标。
    mask = target_raw > 5
    if torch.any(mask):
        mse_mask = float((err[mask] ** 2).mean().item())
        mae_mask = float(abs_err[mask].mean().item())
        mape_mask = float((100.0 * abs_err[mask] / target_raw[mask].abs().clamp_min(1e-6)).mean().item())
    else:
        mse_mask = float('nan')
        mae_mask = float('nan')
        mape_mask = float('nan')

    # normalized_metrics 是原 Metrics(max_min) 的输出，优先用它保证与控制台指标口径完全一致。
    mse_n, mae_n, mse_mask_n, mae_mask_n, mape_mask_n = [float(x) for x in normalized_metrics]
    return {
        'MSE': mse_n if np.isfinite(mse_n) else mse,
        'RMSE': math.sqrt(max(mse_n if np.isfinite(mse_n) else mse, 0.0)),
        'MAE': mae_n if np.isfinite(mae_n) else mae,
        'MSE_Mask': mse_mask_n if np.isfinite(mse_mask_n) else mse_mask,
        'RMSE_Mask': math.sqrt(max(mse_mask_n if np.isfinite(mse_mask_n) else mse_mask, 0.0)),
        'MAE_Mask': mae_mask_n if np.isfinite(mae_mask_n) else mae_mask,
        'MAPE_Mask': mape_mask_n if np.isfinite(mape_mask_n) else mape_mask,
    }


def _daily_metrics(predict_raw, target_raw):
    # [sample, O, D, day] -> 每一天汇总全部 sample/OD
    err = predict_raw - target_raw
    rows = []
    for day in range(err.shape[-1]):
        e = err[..., day].reshape(-1)
        truth = target_raw[..., day].reshape(-1)
        abs_e = e.abs()
        mse = float((e ** 2).mean().item())
        mae = float(abs_e.mean().item())
        valid = truth > 5
        mape = float((100.0 * abs_e[valid] / truth[valid].abs().clamp_min(1e-6)).mean().item()) if torch.any(valid) else float('nan')
        rows.append((mse, math.sqrt(max(mse, 0.0)), mae, mape))
    return rows


def _save_matrix_txt(tensor, path, city_list, dates):
    array = tensor.detach().cpu().numpy()  # [sample, O, D, day]
    samples, n_origin, n_destination, n_days = array.shape
    if len(dates) != n_days:
        dates = [f'Day_{i + 1}' for i in range(n_days)]

    matrix = array.reshape(samples * n_origin * n_destination, n_days)
    labels = []
    for sample in range(samples):
        for i in range(n_origin):
            for j in range(n_destination):
                od = f'{city_list[i]},{city_list[j]}'
                labels.append(od if samples == 1 else f'Sample_{sample + 1}|{od}')

    df = pd.DataFrame(matrix, columns=dates)
    df.insert(0, 'OD', labels)
    df.to_csv(path, sep='\t', index=False)


def Test_and_Save(model, para, dataset, metrics_fn, summary, max_min,
                  city_list, test_dates, batch_size=32, num_workers=0, device='cuda'):
    """加载验证集选出的 BestModel.pt，只对最佳超参数组合测试一次并输出到原 Run_N。"""
    dataloader = _make_dataloaders(dataset, batch_size, num_workers, device)
    model, state = _load_checkpoint(model, summary['checkpoint_path'], device)
    predict_norm, target_norm = _collect_test_predictions(model, dataloader['test'], device)

    normalized_metrics = metrics_fn(predict_norm, target_norm)
    predict_raw = denormalize_tensor(predict_norm, max_min)
    target_raw = denormalize_tensor(target_norm, max_min)
    test_metrics = _raw_test_metrics(predict_raw, target_raw, normalized_metrics)

    run_path = summary['run_path']
    _save_matrix_txt(predict_raw, os.path.join(run_path, 'Predict.txt'), city_list, test_dates)
    _save_matrix_txt(target_raw, os.path.join(run_path, 'Ground Truth.txt'), city_list, test_dates)
    _save_matrix_txt(predict_norm, os.path.join(run_path, 'Predict_Normalized.txt'), city_list, test_dates)

    daily_rows = _daily_metrics(predict_raw, target_raw)
    pd.DataFrame(daily_rows, columns=['MSE', 'RMSE', 'MAE', 'MAPE_Mask'], index=test_dates).rename_axis('Date').reset_index().to_csv(
        os.path.join(run_path, 'Test Daily Metrics.txt'), sep='\t', index=False
    )

    with open(os.path.join(run_path, 'Test Metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('Held-out test set; evaluated only after hyperparameter selection.\n')
        f.write(f"Best validation epoch: {state['epoch']}\n")
        for key in ('MSE', 'RMSE', 'MAE', 'MSE_Mask', 'RMSE_Mask', 'MAE_Mask', 'MAPE_Mask'):
            f.write(f'{key}: {test_metrics[key]:.8f}\n')

    summary['test_metrics'] = test_metrics
    summary['end_time'] = time.time()
    write_result_file(summary)

    print('=' * 100)
    print(f"最佳参数测试结果已输出到: {run_path}")
    print(f"测试集: MSE={test_metrics['MSE']:.3f} | RMSE={test_metrics['RMSE']:.3f} | "
          f"MAE={test_metrics['MAE']:.3f} | MAPE_Mask={test_metrics['MAPE_Mask']:.3f}%")
    return test_metrics
