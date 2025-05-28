#!/usr/bin/env python3
"""
快速测试LSTM改进效果的脚本
用于验证改进是否有效减少预测滞后
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

# 简化的测试配置
TEST_CONFIG = {
    'window_size': 96,
    'prediction_steps': 16,
    'hidden_size': 128,
    'num_layers': 2,
    'batch_size': 32,
    'test_samples': 1000
}

def generate_synthetic_wind_data(n_samples=5000, noise_level=0.1):
    """生成合成风电数据用于测试"""
    t = np.linspace(0, n_samples/4, n_samples)  # 15分钟间隔
    
    # 基础周期性模式
    daily_pattern = np.sin(2 * np.pi * t / 96)  # 24小时周期
    weekly_pattern = 0.3 * np.sin(2 * np.pi * t / (96 * 7))  # 周周期
    
    # 随机趋势
    trend = 0.1 * np.sin(2 * np.pi * t / (96 * 3))
    
    # 噪声
    noise = noise_level * np.random.randn(n_samples)
    
    # 组合并归一化到[0,1]
    data = daily_pattern + weekly_pattern + trend + noise
    data = (data - data.min()) / (data.max() - data.min())
    
    return data

def calculate_lag_score(y_true, y_pred):
    """计算预测滞后分数"""
    # 计算互相关
    correlation = np.correlate(y_true, y_pred, mode='full')
    lag = np.argmax(correlation) - len(y_pred) + 1
    
    # 计算滞后分数 (越小越好)
    lag_score = abs(lag) / len(y_pred)
    return lag_score, lag

def evaluate_prediction_quality(y_true, y_pred, step_labels):
    """评估预测质量"""
    results = {}
    
    for i, label in enumerate(step_labels):
        true_step = y_true[:, i]
        pred_step = y_pred[:, i]
        
        mse = mean_squared_error(true_step, pred_step)
        lag_score, lag = calculate_lag_score(true_step, pred_step)
        
        # 趋势准确率
        true_trend = np.diff(true_step) > 0
        pred_trend = np.diff(pred_step) > 0
        trend_acc = np.mean(true_trend == pred_trend) if len(true_trend) > 0 else 0
        
        results[label] = {
            'mse': mse,
            'lag_score': lag_score,
            'lag_steps': lag,
            'trend_accuracy': trend_acc
        }
    
    return results

def create_test_sequences(data, window_size, prediction_steps):
    """创建测试序列"""
    X, y = [], []
    
    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    
    return np.array(X), np.array(y)

def simple_lstm_baseline(X, y):
    """简单LSTM基线模型"""
    # 使用简单的移动平均作为基线
    predictions = []
    
    for i in range(len(X)):
        # 简单策略：使用最后几个值的平均值作为预测
        last_values = X[i, -5:]  # 最后5个值
        avg_value = np.mean(last_values)
        
        # 为所有预测步使用相同的值（这会导致滞后）
        pred = np.full(y.shape[1], avg_value)
        predictions.append(pred)
    
    return np.array(predictions)

def improved_prediction_strategy(X, y):
    """改进的预测策略（模拟改进后的LSTM）"""
    predictions = []
    
    for i in range(len(X)):
        sequence = X[i]
        
        # 计算趋势
        recent_trend = np.mean(np.diff(sequence[-10:]))
        
        # 基础预测值
        base_value = sequence[-1]
        
        # 为不同步长应用不同的趋势权重
        pred = []
        for step in range(y.shape[1]):
            # 随着预测步长增加，更多地考虑趋势
            trend_weight = 0.1 + 0.05 * step
            step_pred = base_value + recent_trend * (step + 1) * trend_weight
            pred.append(step_pred)
        
        predictions.append(pred)
    
    return np.array(predictions)

def run_comparison_test():
    """运行对比测试"""
    print("=== LSTM改进效果快速测试 ===\n")
    
    # 生成测试数据
    print("1. 生成合成风电数据...")
    data = generate_synthetic_wind_data(n_samples=2000)
    
    # 创建序列
    print("2. 创建测试序列...")
    X, y = create_test_sequences(
        data, 
        TEST_CONFIG['window_size'], 
        TEST_CONFIG['prediction_steps']
    )
    
    # 使用最后1000个样本进行测试
    test_size = min(TEST_CONFIG['test_samples'], len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"测试样本数: {test_size}")
    print(f"输入序列长度: {X_test.shape[1]}")
    print(f"预测步数: {y_test.shape[1]}")
    
    # 预测步标签
    step_labels = ['15min', '30min', '45min', '1h', '1h15min', '1h30min', '1h45min', '2h',
                   '2h15min', '2h30min', '2h45min', '3h', '3h15min', '3h30min', '3h45min', '4h']
    
    # 基线预测
    print("\n3. 运行基线模型（模拟原始LSTM）...")
    baseline_pred = simple_lstm_baseline(X_test, y_test)
    baseline_results = evaluate_prediction_quality(y_test, baseline_pred, step_labels)
    
    # 改进预测
    print("4. 运行改进模型（模拟改进LSTM）...")
    improved_pred = improved_prediction_strategy(X_test, y_test)
    improved_results = evaluate_prediction_quality(y_test, improved_pred, step_labels)
    
    # 结果对比
    print("\n=== 结果对比 ===")
    print(f"{'预测时长':<10} | {'基线MSE':<10} | {'改进MSE':<10} | {'基线滞后':<10} | {'改进滞后':<10} | {'趋势改进':<10}")
    print("-" * 80)
    
    key_steps = [0, 3, 7, 11, 15]  # 15min, 1h, 2h, 3h, 4h
    
    for i in key_steps:
        label = step_labels[i]
        baseline = baseline_results[label]
        improved = improved_results[label]
        
        mse_improvement = (baseline['mse'] - improved['mse']) / baseline['mse'] * 100
        lag_improvement = baseline['lag_score'] - improved['lag_score']
        trend_improvement = improved['trend_accuracy'] - baseline['trend_accuracy']
        
        print(f"{label:<10} | {baseline['mse']:<10.4f} | {improved['mse']:<10.4f} | "
              f"{baseline['lag_score']:<10.4f} | {improved['lag_score']:<10.4f} | "
              f"{trend_improvement:<10.4f}")
    
    # 可视化对比
    print("\n5. 生成对比图...")
    plot_comparison(y_test, baseline_pred, improved_pred, step_labels, key_steps)
    
    return baseline_results, improved_results

def plot_comparison(y_true, baseline_pred, improved_pred, step_labels, key_steps):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 选择前200个样本进行可视化
    n_samples = min(200, len(y_true))
    
    for idx, step_idx in enumerate(key_steps[:4]):
        ax = axes[idx]
        
        true_values = y_true[:n_samples, step_idx]
        baseline_values = baseline_pred[:n_samples, step_idx]
        improved_values = improved_pred[:n_samples, step_idx]
        
        x = range(n_samples)
        
        ax.plot(x, true_values, 'b-', label='真实值', linewidth=1)
        ax.plot(x, baseline_values, 'r--', label='基线预测', alpha=0.7)
        ax.plot(x, improved_values, 'g--', label='改进预测', alpha=0.7)
        
        ax.set_title(f'{step_labels[step_idx]} 预测对比')
        ax.set_xlabel('样本')
        ax.set_ylabel('功率值')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("对比图已保存为 'prediction_comparison.png'")

if __name__ == "__main__":
    baseline_results, improved_results = run_comparison_test()
    
    print("\n=== 总结 ===")
    print("此测试使用合成数据模拟了LSTM改进的效果。")
    print("在实际应用中，改进的LSTM模型应该能够：")
    print("1. 减少长期预测的滞后现象")
    print("2. 提高趋势预测准确性")
    print("3. 降低整体预测误差")
    print("\n请运行改进后的完整LSTM代码以获得真实的改进效果。")
