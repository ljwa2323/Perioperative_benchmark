import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score, 
                            average_precision_score, accuracy_score, recall_score, 
                            precision_score, f1_score, auc)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})  # 修改这里的数字来调整字体大小
plt.rcParams['figure.facecolor'] = 'white'  # 设置画板背景为白色
plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景为白色

# 二分类情况的函数
def read_data_binary(file_list, label_list):
    """
    读取二分类数据
    
    参数:
    file_list: list - 文件路径列表
    label_list: list - 对应的模型标签列表
    
    返回:
    data_dict: dict - 包含每个模型的真实标签和预测概率
    """
    data_dict = {}
    for file, label in zip(file_list, label_list):
        df = pd.read_csv(file)
        data_dict[label] = {
            'y_true': df['y_true'],
            'y_pred': df['y_pred_prob_0']
        }
    return data_dict

def compute_roc_prc_binary(data):
    """
    计算二分类模型的ROC和PRC曲线数据
    
    参数:
    data: dict - 包含每个模型的真实标签和预测概率
    
    返回:
    roc_data: dict - 每个模型的ROC曲线数据(fpr, tpr)
    prc_data: dict - 每个模型的PRC曲线数据(precision, recall)
    """
    roc_data = {}
    prc_data = {}
    
    for model, df in data.items():
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_pred'])
        precision, recall, _ = precision_recall_curve(df['y_true'], df['y_pred'])
        roc_data[model] = (fpr, tpr)
        prc_data[model] = (precision, recall)
        
    return roc_data, prc_data

# 多分类情况的函数
def read_data_multiclass(file_list, label_list, num_classes):
    """
    读取多分类数据
    
    参数:
    file_list: list - 文件路径列表
    label_list: list - 对应的模型标签列表
    num_classes: int - 类别数量
    
    返回:
    data_dict: dict - 包含每个模型的真实标签和预测概率
    """
    data_dict = {}
    for file, label in zip(file_list, label_list):
        df = pd.read_csv(file)
        data_dict[label] = {
            'y_true': df['y_true'],
            'y_pred': df[[f'y_pred_prob_{i}' for i in range(num_classes)]].values
        }
    return data_dict

def compute_roc_prc_multiclass(data):
    """
    计算多分类模型的ROC和PRC曲线数据
    
    参数:
    data: dict - 包含每个模型的真实标签和预测概率
    
    返回:
    roc_data: dict - 每个模型的ROC曲线数据(fpr, tpr)
    prc_data: dict - 每个模型的PRC曲线数据(precision, recall)
    """
    roc_data = {}
    prc_data = {}
    
    for model, df in data.items():
        fpr = {}
        tpr = {}
        precision = {}
        recall = {}
        for i in range(df['y_pred'].shape[1]):  # Assuming one column per class
            fpr[i], tpr[i], _ = roc_curve((df['y_true'] == i).astype(int), df['y_pred'][:, i])
            precision[i], recall[i], _ = precision_recall_curve((df['y_true'] == i).astype(int), df['y_pred'][:, i])
        roc_data[model] = fpr, tpr
        prc_data[model] = precision, recall
        
    return roc_data, prc_data

# 二分类情况的绘图函数
def plot_roc_prc_binary(roc_data, prc_data, data):
    """
    绘制二分类模型的ROC和PRC曲线
    
    参数:
    roc_data: dict - 每个模型的ROC曲线数据
    prc_data: dict - 每个模型的PRC曲线数据
    data: dict - 包含每个模型的真实标签和预测概率
    """
    plt.figure(figsize=(10, 5), dpi=300)

    # ROC Curve
    plt.subplot(1, 2, 1)
    for model, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model} (AUROC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(fontsize='small')  # 调整图例的字体大小
    plt.grid(True)

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for model, (precision, recall) in prc_data.items():
        plt.plot(recall, precision, label=f'{model} (AUPRC = {average_precision_score(data[model]["y_true"], data[model]["y_pred"]):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(fontsize='small')  # 调整图例的字体大小
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 多分类情况的绘图函数
def plot_roc_prc_multiclass(roc_data, prc_data, num_classes, data, class_name):
    """
    绘制多分类模型的ROC和PRC曲线
    
    参数:
    roc_data: dict - 每个模型的ROC曲线数据
    prc_data: dict - 每个模型的PRC曲线数据
    num_classes: int - 类别数量
    data: dict - 包含每个模型的真实标签和预测概率
    class_name: list - 类别名称列表
    """
    plt.figure(figsize=(10, num_classes * 6))  # 调整图形大小以适应所有子图

    # 绘制每个类别的 ROC 和 PRC
    for i in range(num_classes):
        # ROC Curve for each class
        plt.subplot(num_classes + 1, 2, 2*i+1)  # 注意这里的改动，增加了额外的行
        for model, (fpr, tpr) in roc_data.items():
            plt.plot(fpr[i], tpr[i], label=f'{model} (AUROC = {auc(fpr[i], tpr[i]):.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve  ({class_name[i]})')
        plt.legend(fontsize='small')  # 调整图例的字体大小
        plt.grid(True)

        # Precision-Recall Curve for each class
        plt.subplot(num_classes + 1, 2, 2*i+2)
        for model, (precision, recall) in prc_data.items():
            plt.plot(recall[i], precision[i], label=f'{model} (AUPRC = {average_precision_score((data[model]["y_true"] == i).astype(int), data[model]["y_pred"][:, i]):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({class_name[i]})')
        plt.legend(fontsize='small')  # 调整图例的字体大小
        plt.grid(True)

    # 绘制宏平均的 ROC
    plt.subplot(num_classes + 1, 2, num_classes * 2 + 1)
    for model, (fpr, tpr) in roc_data.items():
        # 计算宏平均的 FPR, TPR
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for i in range(num_classes)], axis=0)
        plt.plot(mean_fpr, mean_tpr, label=f'{model} (macro AUC = {auc(mean_fpr, mean_tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average ROC Curve')
    plt.legend(fontsize='small')  # 调整图例的字体大小
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 二分类情况的 compile_results 函数，包括 bootstrap 置信区间
def compile_results_binary(data, n_bootstraps=1000, seed=123):
    """
    计算二分类模型的各项指标及其置信区间
    
    参数:
    data: dict - 包含每个模型的真实标签和预测概率
    n_bootstraps: int - bootstrap采样次数
    seed: int - 随机种子
    
    返回:
    results: dict - 包含每个模型各项指标及其置信区间
    """
    results = {}
    for model, df in data.items():
        try:
            # 初始化存储bootstrap结果的列表
            bootstrapped_auroc = []
            bootstrapped_auprc = []
            bootstrapped_accuracy = []
            bootstrapped_recall = []  # sensitivity
            bootstrapped_specificity = []  # 新增
            bootstrapped_precision = []
            bootstrapped_f1 = []
            
            # 设置随机种子确保可重复性
            np.random.seed(seed)

            # Bootstrap 循环
            for _ in range(n_bootstraps):
                try:
                    indices = resample(np.arange(len(df['y_true'])))
                    y_true_boot = df['y_true'].iloc[indices]
                    y_pred_boot = df['y_pred'].iloc[indices]

                    # 确保bootstrap样本中包含两个类别
                    if len(np.unique(y_true_boot)) < 2:
                        continue

                    # 找到最优阈值
                    precision, recall, thresholds = precision_recall_curve(y_true_boot, y_pred_boot)
                    f1_scores = np.where((recall + precision) > 0, 
                                       2 * recall * precision / (recall + precision), 0)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0
                    y_pred_optimal = (y_pred_boot > optimal_threshold).astype(int)

                    # 计算各项指标
                    bootstrapped_auroc.append(roc_auc_score(y_true_boot, y_pred_boot))
                    bootstrapped_auprc.append(average_precision_score(y_true_boot, y_pred_boot))
                    bootstrapped_accuracy.append(accuracy_score(y_true_boot, y_pred_optimal))
                    bootstrapped_recall.append(recall_score(y_true_boot, y_pred_optimal))
                    # 计算 specificity (true negative rate)
                    tn = np.sum((y_true_boot == 0) & (y_pred_optimal == 0))
                    fp = np.sum((y_true_boot == 0) & (y_pred_optimal == 1))
                    bootstrapped_specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    bootstrapped_precision.append(precision_score(y_true_boot, y_pred_optimal))
                    bootstrapped_f1.append(f1_score(y_true_boot, y_pred_optimal))

                except Exception as e:
                    print(f"警告: Bootstrap样本 {_} 计算出错: {str(e)}")
                    continue

            # 使用完整数据集计算指标
            y_true = df['y_true']
            y_pred = df['y_pred']
            
            if len(np.unique(y_true)) < 2:
                print(f"警告: {model} 只包含一个类别，跳过计算")
                continue
                
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            f1_scores = np.where((recall + precision) > 0, 
                               2 * recall * precision / (recall + precision), 0)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0
            y_pred_optimal = (y_pred > optimal_threshold).astype(int)

            # 计算完整数据集的 specificity
            tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
            fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            results[model] = {
                'AUROC': (roc_auc_score(y_true, y_pred), 
                         np.percentile(bootstrapped_auroc, [2.5, 97.5])),
                'AUPRC': (average_precision_score(y_true, y_pred), 
                         np.percentile(bootstrapped_auprc, [2.5, 97.5])),
                'Accuracy': (accuracy_score(y_true, y_pred_optimal), 
                           np.percentile(bootstrapped_accuracy, [2.5, 97.5])),
                'Sensitivity': (recall_score(y_true, y_pred_optimal), 
                              np.percentile(bootstrapped_recall, [2.5, 97.5])),
                'Specificity': (specificity,
                              np.percentile(bootstrapped_specificity, [2.5, 97.5])),
                'Precision': (precision_score(y_true, y_pred_optimal), 
                            np.percentile(bootstrapped_precision, [2.5, 97.5])),
                'F1': (f1_score(y_true, y_pred_optimal), 
                      np.percentile(bootstrapped_f1, [2.5, 97.5]))
            }

        except Exception as e:
            print(f"警告: 模型 {model} 计算出错: {str(e)}")
            continue

    return results

# 多分类情况的 compile_results 函数，包括 bootstrap 置信区间
def compile_results_multiclass(data, num_classes, n_bootstraps=1000, seed=123):
    """
    计算多分类模型的各项指标及其置信区间
    
    参数:
    data: dict - 包含每个模型的真实标签和预测概率
    num_classes: int - 类别数量
    n_bootstraps: int - bootstrap采样次数
    seed: int - 随机种子
    
    返回:
    results: dict - 包含每个模型各项指标及其置信区间
    """
    results = {}
    for model, df in data.items():
        try:
            model_results = {}
            # 初始化宏平均的指标列表
            macro_scores = {
                'AUROC': [],
                'AUPRC': [],
                'Accuracy': [],
                'Sensitivity': [],  # 改名以保持一致性
                'Specificity': [],  # 新增
                'Precision': [],
                'F1': []
            }
            
            # 设置随机种子确保可重复性
            np.random.seed(seed)

            for i in range(num_classes):
                try:
                    # Bootstrap
                    bootstrapped_scores = {
                        'AUROC': [],
                        'AUPRC': [],
                        'Accuracy': [],
                        'Sensitivity': [],  # 改名以保持一致性
                        'Specificity': [],  # 新增
                        'Precision': [],
                        'F1': []
                    }

                    for _ in range(n_bootstraps):
                        try:
                            indices = resample(np.arange(len(df['y_true'])))
                            y_true_boot = df['y_true'][indices]
                            y_pred_boot = df['y_pred'][indices, i]

                            if len(np.unique(y_true_boot)) < 2:
                                continue

                            # Find the optimal threshold
                            precision, recall, thresholds = precision_recall_curve((y_true_boot == i).astype(int), y_pred_boot)
                            f1_scores = np.where((recall + precision) > 0, 2 * recall * precision / (recall + precision), 0)
                            optimal_idx = np.argmax(f1_scores)
                            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0

                            y_pred_optimal = (y_pred_boot > optimal_threshold).astype(int)
                            y_true_binary = (y_true_boot == i).astype(int)

                            # 计算 specificity
                            tn = np.sum((y_true_binary == 0) & (y_pred_optimal == 0))
                            fp = np.sum((y_true_binary == 0) & (y_pred_optimal == 1))
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                            bootstrapped_scores['AUROC'].append(roc_auc_score(y_true_binary, y_pred_boot))
                            bootstrapped_scores['AUPRC'].append(average_precision_score(y_true_binary, y_pred_boot))
                            bootstrapped_scores['Accuracy'].append(accuracy_score(y_true_binary, y_pred_optimal))
                            bootstrapped_scores['Sensitivity'].append(recall_score(y_true_binary, y_pred_optimal))
                            bootstrapped_scores['Specificity'].append(specificity)
                            bootstrapped_scores['Precision'].append(precision_score(y_true_binary, y_pred_optimal))
                            bootstrapped_scores['F1'].append(f1_score(y_true_binary, y_pred_optimal))

                        except Exception as e:
                            print(f"警告: 模型 {model} 类别 {i} bootstrap {_} 计算出错: {str(e)}")
                            continue

                    # Calculate metrics using the entire dataset
                    y_true = (df['y_true'] == i).astype(int)
                    y_pred = df['y_pred'][:, i]

                    if len(np.unique(y_true)) < 2:
                        print(f"警告: 模型 {model} 类别 {i} 只包含一个类别，跳过计算")
                        continue

                    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                    f1_scores = np.where((recall + precision) > 0, 2 * recall * precision / (recall + precision), 0)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0
                    y_pred_optimal = (y_pred > optimal_threshold).astype(int)

                    # 计算完整数据集的 specificity
                    tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
                    fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    if len(bootstrapped_scores['AUROC']) > 0:
                        class_results = {
                            'AUROC': (roc_auc_score(y_true, y_pred), 
                                    np.percentile(bootstrapped_scores['AUROC'], [2.5, 97.5])),
                            'AUPRC': (average_precision_score(y_true, y_pred), 
                                    np.percentile(bootstrapped_scores['AUPRC'], [2.5, 97.5])),
                            'Accuracy': (accuracy_score(y_true, y_pred_optimal), 
                                       np.percentile(bootstrapped_scores['Accuracy'], [2.5, 97.5])),
                            'Sensitivity': (recall_score(y_true, y_pred_optimal), 
                                          np.percentile(bootstrapped_scores['Sensitivity'], [2.5, 97.5])),
                            'Specificity': (specificity,
                                          np.percentile(bootstrapped_scores['Specificity'], [2.5, 97.5])),
                            'Precision': (precision_score(y_true, y_pred_optimal), 
                                        np.percentile(bootstrapped_scores['Precision'], [2.5, 97.5])),
                            'F1': (f1_score(y_true, y_pred_optimal), 
                                  np.percentile(bootstrapped_scores['F1'], [2.5, 97.5]))
                        }
                        model_results[f'Class {i}'] = class_results

                        # 累加宏平均的指标
                        for key in macro_scores:
                            macro_scores[key].append(class_results[key][0])

                except Exception as e:
                    print(f"警告: 模型 {model} 类别 {i} 计算出错: {str(e)}")
                    continue

            # 只有当至少有一个类别成功计算时才计算宏平均
            if macro_scores['AUROC']:
                macro_results = {
                    metric: (np.mean(values), np.percentile(values, [2.5, 97.5]))
                    for metric, values in macro_scores.items()
                }
                model_results['Macro Average'] = macro_results

            results[model] = model_results

        except Exception as e:
            print(f"警告: 模型 {model} 整体计算出错: {str(e)}")
            continue

    return results

def plot_single_metric_forest(df, metric_col, row_height=0.5, figsize=None, 
                              xlabel=None, title=None, model_col='model',
                              colors=None, marker_size=50, line_width=2):
    """
    绘制单个指标的森林图(Forest Plot)
    
    参数:
    df: DataFrame - 包含模型指标的数据框，格式必须包含模型列及指定的指标列
                   指标格式为："0.8765 (0.8123, 0.9321)"
    metric_col: str - 要绘制的指标列名，如'AUROC'或'AUPRC'等
    row_height: float - 每行的高度
    figsize: tuple - 图形大小，默认根据模型数量自动计算
    xlabel: str - x轴标签，默认使用metric_col的值
    title: str - 图表标题，默认为None
    model_col: str - 包含模型名称的列名，默认为'model'
    colors: dict - 自定义每个模型的颜色，格式为{模型名: 颜色}
    marker_size: int - 点估计值的标记大小
    line_width: int - 置信区间线的宽度
    
    返回:
    fig, ax - matplotlib的图形和轴对象
    """
    # 辅助函数：从格式化字符串中提取数值
    def extract_values(formatted_str):
        try:
            parts = formatted_str.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            point = float(parts[0])
            low = float(parts[1])
            high = float(parts[2])
            return point, low, high
        except Exception as e:
            print(f"提取值出错: {str(e)}")
            return np.nan, np.nan, np.nan
    
    # 设置x轴标签
    if xlabel is None:
        xlabel = metric_col
    
    # 创建绘图数据结构
    processed_data = []
    for _, row in df.iterrows():
        if metric_col not in row:
            print(f"警告: 列 '{metric_col}' 在数据框中不存在")
            continue
            
        model_name = row[model_col]
        point, low, high = extract_values(row[metric_col])
        
        processed_data.append({
            'model': model_name,
            'point': point,
            'low': low,
            'high': high
        })
    
    # 转换为DataFrame
    plot_df = pd.DataFrame(processed_data)
    
    if len(plot_df) == 0:
        print("错误: 无法提取有效数据进行绘图")
        return None, None
    
    # 根据模型数量设置图形大小
    if figsize is None:
        figsize = (8, len(df) * row_height)  # 单一指标可以使用更窄的图形
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取模型及其位置
    models = plot_df['model'].values
    y_positions = np.arange(len(models))
    
    # 计算指标的最小/最大值
    metric_min = plot_df['low'].min()
    metric_max = plot_df['high'].max()
    
    # 设置Y轴刻度和标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    
    # 绘制每个模型的森林图点和线
    for i, (_, data) in enumerate(plot_df.iterrows()):
        # 设置模型颜色
        color = colors.get(data['model'], 'grey') if colors else 'grey'
        
        # 绘制置信区间线
        ax.plot([data['low'], data['high']], [y_positions[i], y_positions[i]], 
                 color=color, alpha=0.7, linewidth=line_width)
        # 绘制点估计值
        ax.scatter(data['point'], y_positions[i], color='black', s=marker_size, zorder=5)
    
    # 设置坐标轴范围和标签
    margin = (metric_max - metric_min) * 0.1
    ax.set_xlim(max(0, metric_min - margin), min(1, metric_max + margin))
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 确保x轴刻度标签保留两位小数
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # 设置Y轴标签方向和位置
    plt.setp(ax.get_yticklabels(), ha="right")
    
    # 固定绘图区域，防止因Y轴标签长度而变化
    fig.subplots_adjust(left=0.3)  # 为Y轴标签预留固定空间
    
    # 设置标题
    if title:
        ax.set_title(title, fontsize=14)
    
    # 紧凑排版后会覆盖我们设置的左侧空间，所以要在tight_layout之后再次调整
    plt.tight_layout()
    fig.subplots_adjust(left=0.3)  # 再次设置左侧边距
    
    return fig, ax


# 辅助函数：计算二分类指标
def calculate_metrics_binary(data, y_true, y_score):
    """
    计算二分类模型的AUROC和AUPRC
    
    参数:
    data: DataFrame - 包含真实标签和预测概率的数据
    y_true: str - 真实标签列名
    y_score: str - 预测概率列名
    
    返回:
    (auroc, auprc): tuple - AUROC和AUPRC值
    """
    try:
        # 检查是否有足够的类别
        if len(np.unique(data[y_true])) < 2:
            print("警告: 只有一个类别，无法计算指标")
            return None, None
            
        # 对于二分类，y_score 应该是正类的预测概率
        auroc = roc_auc_score(data[y_true], data[y_score])
        auprc = average_precision_score(data[y_true], data[y_score])
        
        return auroc, auprc
    except Exception as e:
        print(f"计算二分类指标时出错: {str(e)}")
        return None, None

# 辅助函数：计算二分类指标并带置信区间
def calculate_metrics_binary_with_confidence(data, y_true, y_score, n_bootstraps=1000):
    """
    计算二分类模型的AUROC和AUPRC及其置信区间
    
    参数:
    data: DataFrame - 包含真实标签和预测概率的数据
    y_true: str - 真实标签列名
    y_score: str - 预测概率列名
    n_bootstraps: int - bootstrap采样次数
    
    返回:
    (auroc_results, auprc_results): tuple - 包含点估计和置信区间的AUROC和AUPRC结果
    """
    try:
        auroc_scores = []
        auprc_scores = []
        
        # 检查原始数据是否有足够的类别
        if len(np.unique(data[y_true])) < 2:
            print("警告: 原始数据只有一个类别，无法计算指标")
            return ((float('nan'), (float('nan'), float('nan'))), 
                   (float('nan'), (float('nan'), float('nan'))))
        
        for i in range(n_bootstraps):
            try:
                # Bootstrap sample
                indices = np.random.randint(0, len(data), len(data))
                y_true_sample = data[y_true].iloc[indices]
                y_score_sample = data[y_score].iloc[indices]
                
                # 检查bootstrap样本是否有足够的类别
                if len(np.unique(y_true_sample)) < 2:
                    continue
                
                # Calculate AUROC and AUPRC
                auroc_scores.append(roc_auc_score(y_true_sample, y_score_sample))
                auprc_scores.append(average_precision_score(y_true_sample, y_score_sample))
                
            except Exception as e:
                print(f"Bootstrap {i} 计算出错: {str(e)}")
                continue
        
        # 检查是否有足够的bootstrap结果
        if len(auroc_scores) < n_bootstraps * 0.5:  # 如果成功的bootstrap少于一半
            print(f"警告: 只有 {len(auroc_scores)}/{n_bootstraps} 个bootstrap样本成功计算")
            
        if not auroc_scores:  # 如果没有任何成功的bootstrap
            return None, None
            
        # Calculate mean and 95% confidence intervals
        auroc_mean, auroc_conf = np.mean(auroc_scores), np.percentile(auroc_scores, [2.5, 97.5])
        auprc_mean, auprc_conf = np.mean(auprc_scores), np.percentile(auprc_scores, [2.5, 97.5])
        
        return (round(auroc_mean, 3), auroc_conf.round(3)), (round(auprc_mean, 3), auprc_conf.round(3))
    except Exception as e:
        print(f"计算置信区间时出错: {str(e)}")
        return ((float('nan'), (float('nan'), float('nan'))), 
               (float('nan'), (float('nan'), float('nan'))))

# 辅助函数：计算多分类指标
def calculate_metrics(data, y_true, y_scores):
    """
    计算多分类模型的AUROC和AUPRC
    
    参数:
    data: DataFrame - 包含真实标签和预测概率的数据
    y_true: str - 真实标签列名
    y_scores: list - 预测概率列名列表
    
    返回:
    (auroc, auprc): tuple - AUROC和AUPRC值
    """
    try:
        # 二值化 y_true 以适应多类别情况
        classes = sorted(data[y_true].unique())
        y_true_binarized = label_binarize(data[y_true], classes=classes)
        y_score_matrix = data[y_scores].values
        
        # 检查是否每个类别都至少有一个样本
        if not all(np.sum(y_true_binarized, axis=0) > 0):
            print("警告: 某些类别没有样本")
            return None, None
            
        auroc = roc_auc_score(y_true_binarized, y_score_matrix, multi_class='ovr')
        
        # 计算每个类别的 AUPRC 并取平均
        auprc_scores = []
        for i in range(y_score_matrix.shape[1]):
            auprc_scores.append(average_precision_score(y_true_binarized[:, i], y_score_matrix[:, i]))
        auprc = sum(auprc_scores) / len(auprc_scores)
        
        return auroc, auprc
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None, None

# 辅助函数：计算多分类指标并带置信区间
def calculate_metrics_with_confidence(data, y_true, y_scores, n_bootstraps=1000):
    """
    计算多分类模型的AUROC和AUPRC及其置信区间
    
    参数:
    data: DataFrame - 包含真实标签和预测概率的数据
    y_true: str - 真实标签列名
    y_scores: list - 预测概率列名列表
    n_bootstraps: int - bootstrap采样次数
    
    返回:
    (auroc_results, auprc_results): tuple - 包含点估计和置信区间的AUROC和AUPRC结果
    """
    try:
        classes = sorted(data[y_true].unique())
        y_true_binarized = label_binarize(data[y_true], classes=classes)
        y_score_matrix = data[y_scores].values
        
        # 检查原始数据是否每个类别都有样本
        if not all(np.sum(y_true_binarized, axis=0) > 0):
            print("警告: 某些类别在原始数据中没有样本")
            return None, None
            
        auroc_scores = []
        auprc_scores = []
        
        for i in range(n_bootstraps):
            try:
                # Bootstrap sample
                indices = np.random.randint(0, len(data), len(data))
                y_true_sample = y_true_binarized[indices]
                y_score_sample = y_score_matrix[indices]
                
                # 检查bootstrap样本是否每个类别都有样本
                if not all(np.sum(y_true_sample, axis=0) > 0):
                    continue
                
                # Calculate AUROC and AUPRC for each class and average
                auroc = roc_auc_score(y_true_sample, y_score_sample, multi_class='ovr')
                auroc_scores.append(auroc)
                
                class_auprc_scores = []
                for j in range(y_score_sample.shape[1]):
                    class_auprc_scores.append(average_precision_score(y_true_sample[:, j], y_score_sample[:, j]))
                auprc_scores.append(np.mean(class_auprc_scores))
                
            except Exception as e:
                print(f"Bootstrap {i} 计算出错: {str(e)}")
                continue
        
        # 检查是否有足够的bootstrap结果
        if len(auroc_scores) < n_bootstraps * 0.5:
            print(f"警告: 只有 {len(auroc_scores)}/{n_bootstraps} 个bootstrap样本成功计算")
            
        if not auroc_scores:
            return None, None
        
        # Calculate mean and 95% confidence intervals
        auroc_mean, auroc_conf = np.mean(auroc_scores), np.percentile(auroc_scores, [2.5, 97.5])
        auprc_mean, auprc_conf = np.mean(auprc_scores), np.percentile(auprc_scores, [2.5, 97.5])
        
        return (round(auroc_mean, 3), auroc_conf.round(3)), (round(auprc_mean, 3), auprc_conf.round(3))
    except Exception as e:
        print(f"计算多分类指标置信区间时出错: {str(e)}")
        return None, None

# 示例用法函数
def example_binary_metrics_workflow(root_path, file_list, label_list, n_bootstraps=50):
    """
    二分类模型评估工作流示例
    
    参数:
    root_path: str - 数据根目录
    file_list: list - 文件名列表
    label_list: list - 对应的模型标签列表
    n_bootstraps: int - bootstrap采样次数
    
    返回:
    无，直接显示结果图表
    """
    # 完整文件路径
    full_file_list = [os.path.join(root_path, f) for f in file_list]
    
    # 读取数据并计算指标
    datas = read_data_binary(full_file_list, label_list)
    roc_data, prc_data = compute_roc_prc_binary(datas)
    results = compile_results_binary(datas, n_bootstraps=n_bootstraps)
    
    # 将结果字典转换为DataFrame
    data_rows = []
    for model, metrics in results.items():
        row = {'model': model}
        for metric, (point_estimate, ci) in metrics.items():
            row[metric] = f"{point_estimate:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
        data_rows.append(row)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(data_rows)
    results_df = results_df[['model'] + [col for col in results_df if col != 'model']]
    
    # 绘制ROC和PRC曲线
    plot_roc_prc_binary(roc_data, prc_data, datas)
    
    # 绘制模型性能对比的森林图
    fig, axes = plot_metric_forest(
        results_df, 
        metrics=['AUROC', 'AUPRC', 'Sensitivity', 'Specificity'],
        row_height=0.4
    )
    
    # 显示图像
    plt.show()
    
    return results_df, fig