import torch
import copy

import os
import numpy as np
from datetime import datetime

import pandas as pd
from scipy import stats

def calculate_integrated_gradients(input_list, model, target_class_index, steps=50, device='cuda:0', baseline_list=None):
    # 将所有的 Tensor 移到 GPU 上并初始化基线和梯度向量
    for i in range(len(input_list)):
        input_list[i] = input_list[i].to(device)

    if baseline_list is None:
        baseline_list = [torch.zeros_like(input_tensor, device=device) for input_tensor in input_list]
    else:
        # 根据 input_list 的形状扩展 baseline_list
        baseline_list = [baseline.to(device).expand_as(input_tensor) for baseline, input_tensor in zip(baseline_list, input_list)]

    integrated_gradients_list = [torch.zeros_like(input_tensor) for input_tensor in input_list]
    diff_list = [(input_tensor - baseline_tensor) / steps for input_tensor, baseline_tensor in zip(input_list, baseline_list)]

    # 对于每个步骤，计算模型的预测和梯度
    for step in range(steps):
        for i in range(len(input_list)):
            # 创建当前步骤的输入列表，其中只有第i个元素是变化的
            current_step_input = input_list[:]  # 使用原始输入列表的副本
            current_step_input[i] = baseline_list[i] + step * diff_list[i]  # 只更新第i个元素
            current_step_input[i].requires_grad_()  # 为当前变量启用梯度跟踪

            # 执行模型前向传播
            preds = model(current_step_input)
            target_preds = preds[0][:, target_class_index]  # 选择目标类别的预测值

            # 计算梯度
            grads = torch.autograd.grad(target_preds, current_step_input[i], retain_graph=True)[0]

            # 检查当前步骤的输入是否与基线相同，如果相同，则梯度设置为零
            mask = (torch.abs(current_step_input[i] - baseline_list[i]) < 1e-6).float()  # 修改
            grads = grads * (1 - mask)

            integrated_gradients_list[i] += grads

    # 将计算的梯度除以步数并转移到 CPU
    for i in range(len(integrated_gradients_list)):
        integrated_gradients_list[i] = (integrated_gradients_list[i] / steps).to('cpu')

    return integrated_gradients_list

class Logger:
    """
    简易训练日志记录器，用于跟踪训练过程并保存模型检查点
    """
    def __init__(self, save_dir='checkpoints', model_name='model', resume=False, use_timestamp=False):
        """
        初始化 Logger
        
        参数:
            save_dir (str): 保存检查点的目录
            model_name (str): 模型名称，用于命名保存的文件
            resume (bool): 是否从上次训练中恢复
            use_timestamp (bool): 是否在文件夹名称中包含时间戳
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_auc = 0.0
        self.best_epoch = -1
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.use_timestamp = use_timestamp
        
        if resume:
            # 查找最近的训练目录
            model_dirs = [d for d in os.listdir(save_dir) if d.startswith(model_name)]
            if model_dirs:
                # 按时间戳排序，获取最近的目录
                model_dirs.sort(reverse=True)
                self.log_dir = os.path.join(save_dir, model_dirs[0])
                print(f"继续训练，使用目录: {self.log_dir}")
                
                # 从日志文件中恢复训练历史
                self.log_file = os.path.join(self.log_dir, 'training_log.txt')
                if os.path.exists(self.log_file):
                    self._load_training_history()
            else:
                print("未找到之前的训练记录，将开始新的训练")
                self._create_new_log_dir()
        else:
            # 创建新的保存目录
            self._create_new_log_dir()
    
    def _create_new_log_dir(self):
        """创建新的日志目录"""
        if self.use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_dir = os.path.join(self.save_dir, f"{self.model_name}_{timestamp}")
        else:
            self.log_dir = os.path.join(self.save_dir, f"{self.model_name}")
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = os.path.join(self.log_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,valid_loss,auc,is_best\n")
    
    def _load_training_history(self):
        """从日志文件加载训练历史"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[1:]  # 跳过标题行
                if lines:
                    for line in lines:
                        epoch, train_loss, valid_loss, auc, is_best = line.strip().split(',')
                        epoch = int(epoch)
                        auc = float(auc)
                        train_loss = float(train_loss)
                        
                        # 更新最后一个 epoch 和最佳指标
                        self.start_epoch = epoch + 1
                        
                        if is_best.lower() == 'true':
                            self.best_epoch = epoch
                            self.best_auc = auc
                            self.best_loss = train_loss
                    
                    print(f"恢复训练历史: 开始轮次 {self.start_epoch}, 最佳 AUC: {self.best_auc:.4f} (轮次 {self.best_epoch})")
        except Exception as e:
            print(f"加载训练历史时出错: {e}")
            print("将开始新的训练")
            self.start_epoch = 0
            self.best_auc = 0.0
            self.best_epoch = -1
            self.best_loss = float('inf')
    
    def update_best_metrics(self, epoch, train_loss, auc):
        """
        更新最佳指标
        
        参数:
            epoch (int): 当前训练轮次
            train_loss (float): 训练损失
            auc (float): AUC 值
            
        返回:
            bool: 如果当前模型是最佳模型则返回 True
        """
        is_best = auc > self.best_auc
        
        if is_best:
            self.best_auc = auc
            self.best_epoch = epoch
            self.best_loss = train_loss
            
        # 记录到日志文件
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{train_loss},{auc},{is_best}\n")
            
        return is_best
    
    def load_latest_checkpoint(self, model, optimizer=None):
        """
        加载最新的检查点以继续训练
        
        参数:
            model (nn.Module): 要加载权重的模型
            optimizer (Optimizer, optional): 优化器，用于恢复训练状态
            
        返回:
            tuple: (model, optimizer, start_epoch)
        """
        latest_path = os.path.join(self.log_dir, f"{self.model_name}_latest.pth")
        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"加载最新模型 (epoch {checkpoint['epoch']}, AUC: {checkpoint['auc']:.4f})")
            return model, optimizer, checkpoint['epoch'] + 1
        else:
            print("未找到最新模型文件，将从头开始训练")
            return model, optimizer, 0
    
    def save_checkpoint(self, model, epoch, train_loss, valid_loss, auc, is_best, optimizer=None):
        """
        保存模型检查点
        
        参数:
            model (nn.Module): 要保存的模型
            epoch (int): 当前训练轮次
            train_loss (float): 训练损失
            valid_loss (float): 验证损失
            auc (float): AUC 值
            is_best (bool): 是否是最佳模型
            optimizer (Optimizer, optional): 优化器，用于保存训练状态
        """
        # 保存最新检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'auc': auc,
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch
        }
        
        # 如果提供了优化器，也保存其状态
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存最新模型
        latest_path = os.path.join(self.log_dir, f"{self.model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，单独保存一份
        if is_best:
            best_path = os.path.join(self.log_dir, f"{self.model_name}_best.pth")
            torch.save(checkpoint, best_path)
            
            # 保存当前轮次的模型
            epoch_path = os.path.join(self.log_dir, f"{self.model_name}_epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_path)
    
    def load_best_model(self, model):
        """
        加载最佳模型
        
        参数:
            model (nn.Module): 要加载权重的模型
            
        返回:
            nn.Module: 加载了最佳权重的模型
        """
        best_path = os.path.join(self.log_dir, f"{self.model_name}_best.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型 (epoch {checkpoint['epoch']}, AUC: {checkpoint['auc']:.4f})")
        else:
            print("未找到最佳模型文件")
        return model
    
    def get_training_summary(self):
        """
        获取训练摘要信息
        
        返回:
            str: 训练摘要信息
        """
        return f"最佳模型: Epoch {self.best_epoch}, AUC: {self.best_auc:.4f}, Loss: {self.best_loss:.6f}"


def calculate_statistics(data, var_dict, data_format='wide'):
    """
    根据变量字典计算指定的统计量
    
    参数:
        data (pd.DataFrame): 输入数据，可以是长表或宽表格式
        var_dict (pd.DataFrame): 变量字典，包含变量名、类型和需要计算的统计量
        data_format (str): 数据格式，'wide'表示宽表，'long'表示长表
        
    返回:
        pd.DataFrame: 包含所有计算结果的单行数据框
    """
    
    # 初始化结果字典
    results = {}
    
    # 处理长表格式
    if data_format == 'long':
        # 假设长表有var, time, value三列
        for _, row in var_dict.iterrows():
            var_name = row['item_name']
            var_type = row['value_type']
            stats_to_calc = row['stat'].split('|')
            
            # 提取当前变量的数据
            var_data = data[data['var'] == var_name]['value']
            
            if len(var_data) == 0:
                # 如果没有找到变量数据，所有统计量设为NA
                for stat in stats_to_calc:
                    results[f"{var_name}_{stat}"] = np.nan
                continue
            
            # 根据变量类型计算统计量
            _calculate_stats(var_name, var_type, stats_to_calc, var_data, results)
    
    # 处理宽表格式
    else:  # data_format == 'wide'
        for _, row in var_dict.iterrows():
            var_name = row['item_name']
            var_type = row['value_type']
            stats_to_calc = row['stat'].split('|')
            
            # 检查变量是否在数据中
            if var_name not in data.columns:
                # 如果没有找到变量，所有统计量设为NA
                for stat in stats_to_calc:
                    results[f"{var_name}_{stat}"] = np.nan
                continue
            
            # 提取当前变量的数据
            var_data = data[var_name]
            
            # 根据变量类型计算统计量
            _calculate_stats(var_name, var_type, stats_to_calc, var_data, results)
    
    # 将结果转换为DataFrame并返回
    return pd.DataFrame([results])

def _calculate_stats(var_name, var_type, stats_to_calc, var_data, results):
    """
    计算指定变量的统计量，自动忽略 NA 值
    
    参数:
        var_name (str): 变量名
        var_type (str): 变量类型 (num/ord/cat/bin)
        stats_to_calc (list): 需要计算的统计量列表
        var_data (pd.Series): 变量数据
        results (dict): 存储结果的字典
    """
    # 数值型变量
    if var_type == 'num':
        for stat in stats_to_calc:
            try:
                # 去除 NA 值
                clean_data = var_data.dropna()
                
                if len(clean_data) == 0:
                    # 如果去除 NA 后没有数据，则设为 NA
                    results[f"{var_name}_{stat}"] = np.nan
                    continue
                    
                if stat == 'mean':
                    results[f"{var_name}_{stat}"] = clean_data.mean()
                elif stat == 'median':
                    results[f"{var_name}_{stat}"] = clean_data.median()
                elif stat == 'sd':
                    results[f"{var_name}_{stat}"] = clean_data.std()
                elif stat == 'max':
                    results[f"{var_name}_{stat}"] = clean_data.max()
                elif stat == 'min':
                    results[f"{var_name}_{stat}"] = clean_data.min()
                elif stat == 'iqr':
                    q75, q25 = np.percentile(clean_data, [75, 25])
                    results[f"{var_name}_{stat}"] = q75 - q25
                elif stat == 'sum':
                    results[f"{var_name}_{stat}"] = clean_data.sum()
                else:
                    results[f"{var_name}_{stat}"] = np.nan
            except:
                results[f"{var_name}_{stat}"] = np.nan
    
    # 有序变量或分类变量
    elif var_type in ['ord', 'cat']:
        for stat in stats_to_calc:
            try:
                # 去除 NA 值
                clean_data = var_data.dropna()
                
                if len(clean_data) == 0:
                    # 如果去除 NA 后没有数据，则设为 NA
                    results[f"{var_name}_{stat}"] = np.nan
                    continue
                    
                if stat == 'mode':
                    # 获取众数（可能有多个）
                    mode_result = stats.mode(clean_data)
                    # 在新版scipy中，mode返回的是ModeResult对象，需要获取mode属性
                    if hasattr(mode_result, 'mode'):
                        mode_value = mode_result.mode[0]
                    else:
                        mode_value = mode_result[0][0]
                    results[f"{var_name}_{stat}"] = mode_value
                # 为有序变量添加额外的统计量计算
                elif var_type == 'ord' and stat == 'median':
                    results[f"{var_name}_{stat}"] = clean_data.median()
                elif var_type == 'ord' and stat == 'max':
                    results[f"{var_name}_{stat}"] = clean_data.max()
                elif var_type == 'ord' and stat == 'min':
                    results[f"{var_name}_{stat}"] = clean_data.min()
                else:
                    results[f"{var_name}_{stat}"] = np.nan
            except:
                results[f"{var_name}_{stat}"] = np.nan
    
    # 二元变量
    elif var_type == 'bin':
        for stat in stats_to_calc:
            try:
                # 去除 NA 值
                clean_data = var_data.dropna()
                
                if len(clean_data) == 0:
                    # 如果去除 NA 后没有数据，则设为 NA
                    results[f"{var_name}_{stat}"] = np.nan
                    continue
                    
                if stat == 'any':
                    # 检查是否有任何True值
                    results[f"{var_name}_{stat}"] = clean_data.any()
                else:
                    results[f"{var_name}_{stat}"] = np.nan
            except:
                results[f"{var_name}_{stat}"] = np.nan
    
    # 未知变量类型
    else:
        for stat in stats_to_calc:
            results[f"{var_name}_{stat}"] = np.nan