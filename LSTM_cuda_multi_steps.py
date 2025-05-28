import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import os
import random
import time
from typing import Union
from datetime import datetime

# I. 项目设置与配置
# 1. 环境初始化 - CUDA GPU支持
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using device: {DEVICE}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print(f"CUDA not available, using device: {DEVICE}")

# 2. 定义全局超参数 - 针对RTX 3050Ti 4GB优化
PREDICTION_STEPS = 16
WINDOW_SIZE = 96  # 增加到24小时历史数据 (96 * 15min = 24h)
LSTM_HIDDEN_SIZE = 256  # 适合4GB显存的隐藏层大小
LSTM_NUM_LAYERS = 2  # 减少LSTM层数以节省显存
LSTM_DROPOUT = 0.3
FC_LAYERS = [128, 64]  # 适合4GB显存的全连接层
BATCH_SIZE = 32  # 适合4GB显存的批次大小
LEARNING_RATE = 0.0005  # 降低学习率
MAX_EPOCHS = 150 # 适中的训练轮数
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0001
NORMALIZATION_METHOD = "standard"  # 改用标准化
MODEL_SAVE_PATH = "./lstm_wpp_model_cuda.pth" # CUDA版本模型保存路径
FIGURE_SAVE_DIR = "./figures_cuda" # CUDA版本图像保存目录
TRAIN_DATA_PATH = "train_set.xlsx"
TEST_DATA_PATH = "test_set.xlsx"

# 3. 设置全局随机种子 - 包含CUDA设置
SEED = 42
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 为了可重现性
        print("CUDA random seeds set for reproducibility")

set_seed(SEED)

# 动态显存检测和参数调整
def adjust_params_for_gpu():
    """根据GPU显存动态调整参数"""
    global LSTM_HIDDEN_SIZE, BATCH_SIZE, FC_LAYERS, LSTM_NUM_LAYERS

    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")

        if gpu_memory_gb <= 4.5:  # RTX 3050Ti 4GB
            LSTM_HIDDEN_SIZE = 256
            BATCH_SIZE = 24  # 进一步减小批次大小
            FC_LAYERS = [128, 64]
            LSTM_NUM_LAYERS = 2
            print("Adjusted parameters for 4GB GPU (RTX 3050Ti)")
        elif gpu_memory_gb <= 6.5:  # RTX 3060 6GB
            LSTM_HIDDEN_SIZE = 384
            BATCH_SIZE = 32
            FC_LAYERS = [192, 96]
            LSTM_NUM_LAYERS = 2
            print("Adjusted parameters for 6GB GPU")
        elif gpu_memory_gb <= 8.5:  # RTX 3070 8GB
            LSTM_HIDDEN_SIZE = 512
            BATCH_SIZE = 48
            FC_LAYERS = [256, 128, 64]
            LSTM_NUM_LAYERS = 3
            print("Adjusted parameters for 8GB GPU")
        else:  # RTX 3080+ 10GB+
            LSTM_HIDDEN_SIZE = 768
            BATCH_SIZE = 64
            FC_LAYERS = [384, 192, 96]
            LSTM_NUM_LAYERS = 3
            print("Adjusted parameters for high-end GPU")

        print(f"Final parameters: Hidden={LSTM_HIDDEN_SIZE}, Batch={BATCH_SIZE}, Layers={LSTM_NUM_LAYERS}")

adjust_params_for_gpu()

# 确保图像保存目录存在
if not os.path.exists(FIGURE_SAVE_DIR):
    os.makedirs(FIGURE_SAVE_DIR)
    print(f"Created directory: {FIGURE_SAVE_DIR}")

# II. 数据处理模块
# 1. 原始数据加载与清洗
def load_and_clean_data(file_path: str) -> pd.Series:
    """
    从指定相对路径的Excel文件（单列、无表头）加载数据到 Pandas Series。
    处理缺失值 (使用前一个有效值填充)。
    确保数据为数值类型 (float)。
    """
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_excel(file_path, header=None, names=['power'])
        data_series = data['power']
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.Series(dtype=float)

    data_series = pd.to_numeric(data_series, errors='coerce')

    if data_series.isnull().any():
        print("Warning: NaNs found after numeric conversion. Applying ffill and bfill.")
        data_series = data_series.ffill()
        data_series = data_series.bfill()

    if data_series.isnull().any():
        print("Warning: NaNs still present after ffill/bfill. Filling remaining NaNs with 0.")
        data_series = data_series.fillna(0)
        if data_series.isnull().all():
             print("Critical Warning: The entire data series was NaN and has been filled with 0.")

    print(f"Data loaded and cleaned. Shape: {data_series.shape}")
    return data_series

# 2. 数据标准化与反标准化
class DataScaler:
    def __init__(self, method="minmax"):
        if method not in ["minmax", "standard"]:
            raise ValueError("Method should be 'minmax' or 'standard'")
        self.method = method
        self.scaler = None
        print(f"DataScaler initialized with method: {self.method}")

    def fit(self, data: Union[pd.Series, np.ndarray]):
        if isinstance(data, pd.Series):
            data_np = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data_np = data.reshape(-1, 1)
            else:
                data_np = data
        else:
            raise TypeError("Input data must be a Pandas Series or NumPy array.")

        if self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(data_np)
        print("Scaler fitted.")

    def transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        if isinstance(data, pd.Series):
            data_np = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data_np = data.reshape(-1, 1)
            else:
                data_np = data
        else:
            raise TypeError("Input data must be a Pandas Series or NumPy array.")

        scaled_data = self.scaler.transform(data_np)
        print(f"Data transformed. Original shape: {data_np.shape}, Scaled shape: {scaled_data.shape}")
        return scaled_data.flatten()

    def fit_transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        if data.ndim == 1:
            data_np = data.reshape(-1, 1)
        elif data.ndim == 2 and data.shape[1] == 1:
            data_np = data
        elif data.ndim == 2 and data.shape[1] > 1 :
            print(f"Warning: inverse_transform called on multi-column data (shape {data.shape}). Assuming scaling on the first column or that all columns use the same scale.")
            data_np = data
        else:
            raise ValueError(f"Input data for inverse_transform has incompatible shape: {data.shape}")

        original_data = self.scaler.inverse_transform(data_np)
        print(f"Data inverse-transformed. Scaled shape: {data_np.shape}, Original shape: {original_data.shape}")
        return original_data

# 3. 序列数据构建
def create_sequences(data: np.ndarray, window_size: int, prediction_steps: int, device: torch.device):
    X_list, y_list = [], []
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - window_size - prediction_steps + 1):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size : i + window_size + prediction_steps].flatten())

    if not X_list:
        print("Warning: Not enough data to create sequences with the given window_size and prediction_steps.")
        return torch.empty(0, window_size, 1, device=device), torch.empty(0, prediction_steps, device=device)

    X = np.array(X_list)
    y = np.array(y_list)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    print(f"Sequences created. X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    return X_tensor, y_tensor

# III. 改进的LSTM模型定义 (PyTorch nn.Module)
class AttentionLayer(nn.Module):
    """简单的注意力机制层"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector: (batch_size, hidden_size)
        return context_vector, attention_weights

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, num_layers: int,
                 dropout_rate: float, fc_layers_config: list, output_steps: int):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps

        # 双向LSTM
        self.lstm = nn.LSTM(input_size=input_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            bidirectional=True)

        # 注意力机制
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional

        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        fc_module_list = []
        current_dim = hidden_size * 2  # *2 for bidirectional
        if fc_layers_config:
            for fc_hidden_dim in fc_layers_config:
                fc_module_list.append(nn.Linear(current_dim, fc_hidden_dim))
                fc_module_list.append(nn.BatchNorm1d(fc_hidden_dim))
                fc_module_list.append(nn.ReLU())
                fc_module_list.append(nn.Dropout(dropout_rate))
                current_dim = fc_hidden_dim
        self.fc_layers = nn.Sequential(*fc_module_list)

        # 分步预测层 - 为不同预测步长使用不同的输出层
        self.step_predictors = nn.ModuleList([
            nn.Linear(current_dim, 1) for _ in range(output_steps)
        ])

        print("ImprovedLSTMModel initialized.")
        print(f"  Input features: {input_features}")
        print(f"  Hidden size: {hidden_size}, Num layers: {num_layers}, Bidirectional: True")
        print(f"  LSTM Dropout: {dropout_rate if num_layers > 1 else 0}")
        print(f"  FC Layers Config: {fc_layers_config}")
        print(f"  Output steps: {output_steps}")
        print(f"  Using attention mechanism and step-specific predictors")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 初始化隐藏状态 (双向LSTM需要 2 * num_layers)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 注意力机制
        context_vector, attention_weights = self.attention(lstm_out)

        # Dropout
        out = self.dropout(context_vector)

        # 全连接层
        if hasattr(self, 'fc_layers') and len(self.fc_layers) > 0:
            out = self.fc_layers(out)

        # 分步预测
        predictions = []
        for i, predictor in enumerate(self.step_predictors):
            step_pred = predictor(out)
            predictions.append(step_pred)

        # 合并所有预测步
        output = torch.cat(predictions, dim=1)  # (batch_size, output_steps)
        return output

# 保持原有模型作为备选
class LSTMModel(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, num_layers: int,
                 dropout_rate: float, fc_layers_config: list, output_steps: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_rate)

        fc_module_list = []
        current_dim = hidden_size
        if fc_layers_config:
            for fc_hidden_dim in fc_layers_config:
                fc_module_list.append(nn.Linear(current_dim, fc_hidden_dim))
                fc_module_list.append(nn.ReLU())
                current_dim = fc_hidden_dim
        self.fc_layers = nn.Sequential(*fc_module_list)

        self.output_layer = nn.Linear(current_dim, output_steps)

        print("LSTMModel initialized.")
        print(f"  Input features: {input_features}")
        print(f"  Hidden size: {hidden_size}, Num layers: {num_layers}, LSTM Dropout: {dropout_rate if num_layers > 1 else 0}")
        print(f"  FC Layers Config: {fc_layers_config}")
        print(f"  Output steps: {output_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()

        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        out = lstm_out[:, -1, :]
        out = self.dropout(out)

        if hasattr(self, 'fc_layers') and len(self.fc_layers) > 0:
            out = self.fc_layers(out)

        out = self.output_layer(out)
        return out

# 改进的损失函数
class WeightedMSELoss(nn.Module):
    """为不同预测步长使用不同权重的MSE损失"""
    def __init__(self, prediction_steps: int, weight_decay: float = 0.9):
        super(WeightedMSELoss, self).__init__()
        self.prediction_steps = prediction_steps
        # 为长期预测分配更高权重以减少滞后
        self.weights = torch.tensor([weight_decay ** i for i in range(prediction_steps)])
        # 归一化权重
        self.weights = self.weights / self.weights.sum()
        print(f"WeightedMSELoss initialized with weights: {self.weights}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: (batch_size, prediction_steps)
        # targets: (batch_size, prediction_steps)
        weights = self.weights.to(predictions.device)

        # 计算每个步长的MSE
        step_losses = torch.mean((predictions - targets) ** 2, dim=0)  # (prediction_steps,)

        # 加权平均
        weighted_loss = torch.sum(step_losses * weights)
        return weighted_loss

# IV. 模型训练与验证模块
# 1. 数据准备
def prepare_dataloaders(train_file: str, val_split_ratio: float, scaler: DataScaler,
                        window_size: int, prediction_steps: int, batch_size: int,
                        device: torch.device):
    print("Preparing dataloaders...")
    raw_data = load_and_clean_data(train_file)
    if raw_data.empty:
        raise ValueError(f"No data loaded from {train_file}. Cannot prepare dataloaders.")

    scaled_data = scaler.fit_transform(raw_data)

    split_index = int(len(scaled_data) * (1 - val_split_ratio))
    train_data_scaled = scaled_data[:split_index]
    val_data_scaled = scaled_data[split_index:]

    print(f"Train data scaled shape: {train_data_scaled.shape}")
    print(f"Validation data scaled shape: {val_data_scaled.shape}")

    if len(train_data_scaled) < window_size + prediction_steps or len(val_data_scaled) < window_size + prediction_steps:
        min_len = window_size + prediction_steps
        print(f"Warning: Train or validation set too small for sequence creation (min length {min_len}).")
        print(f"Train length: {len(train_data_scaled)}, Val length: {len(val_data_scaled)}")
        empty_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)
        return empty_loader, empty_loader

    X_train, y_train = create_sequences(train_data_scaled, window_size, prediction_steps, device)
    X_val, y_val = create_sequences(val_data_scaled, window_size, prediction_steps, device)

    if X_train.nelement() == 0 or X_val.nelement() == 0:
        print("Warning: Sequence creation resulted in empty tensors for train or validation set.")
        empty_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)
        return empty_loader, empty_loader

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train DataLoader: {len(train_loader)} batches, Val DataLoader: {len(val_loader)} batches.")
    return train_loader, val_loader

# 2. 训练主函数
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                max_epochs: int, early_stopping_patience: int,
                early_stopping_min_delta: float, device: torch.device,
                model_save_path: str, scheduler=None):
    print("Starting model training...")
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_temp_path = 'best_model_temp_cuda.pth' # Temp path for CUDA version

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        model.train()
        running_train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch_val, y_batch_val in val_loader:
                X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                outputs_val = model(X_batch_val)
                val_loss = criterion(outputs_val, y_batch_val)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        epoch_duration = time.time() - start_time_epoch

        # GPU内存使用情况监控
        gpu_memory_info = ""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_info = f", GPU Mem: {gpu_memory_allocated:.1f}/{gpu_memory_reserved:.1f}GB"

        print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s{gpu_memory_info}")

        # 学习率调度器步进
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_temp_path)
            print(f"Validation loss improved. Saved best model to {best_model_temp_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            if os.path.exists(best_model_temp_path):
                print(f"Loading best model from {best_model_temp_path}")
                model.load_state_dict(torch.load(best_model_temp_path, map_location=device)) # Ensure map_location for CPU
            break

    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    if os.path.exists(best_model_temp_path):
        os.remove(best_model_temp_path)
        print(f"Removed temporary best model file: {best_model_temp_path}")

    return train_loss_history, val_loss_history

# V. 模型评估与预测模块
# 1. 通用预测函数
def make_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())

    if not all_predictions:
        return np.array([])

    return np.concatenate(all_predictions, axis=0)

# 2. 在指定数据集上进行评估

def calculate_cr_accuracy(P_M, P_P): # noqa: N802
    """
    计算功率预测准确率 C_R。

    Args:
        P_M (ndarray): 实际功率值 (y_true_actual)。
        P_P (ndarray): 预测功率值 (y_pred_actual)。

    Returns:
        float: 功率预测准确率 C_R (百分比)。
    """
    if not isinstance(P_M, np.ndarray):
        P_M = np.array(P_M)
    if not isinstance(P_P, np.ndarray):
        P_P = np.array(P_P)

    if P_M.shape != P_P.shape:
        raise ValueError(f"实际功率和预测功率的形状必须一致。P_M shape: {P_M.shape}, P_P shape: {P_P.shape}")
    if P_M.size == 0:
        return 0.0 # 或者根据需要返回 np.nan 或其他

    N = len(P_M)

    # 初始化 R_values 数组
    R_values = np.zeros_like(P_M, dtype=float)

    # 条件 P_M_i > 0.2
    mask_gt_02 = P_M > 0.2
    # 对于 P_M[mask_gt_02]，由于 P_M > 0.2，分母不会是0
    R_values[mask_gt_02] = (P_M[mask_gt_02] - P_P[mask_gt_02]) / P_M[mask_gt_02]

    # 条件 P_M_i <= 0.2
    mask_le_02 = P_M <= 0.2
    R_values[mask_le_02] = (P_M[mask_le_02] - P_P[mask_le_02]) / 0.2

    R_i_squared_sum = np.sum(R_values**2)

    if N == 0: # 再次检查，以防 P_M 为空数组
        return 0.0

    # 计算 C_R
    # C_R = (1 - sqrt((1/N) * sum(R_i^2))) * 100%
    term_inside_sqrt = R_i_squared_sum / N
    # 处理 term_inside_sqrt 可能为负的极端情况（理论上平方和除以N不会为负，但浮点数精度可能导致微小负值）
    if term_inside_sqrt < 0:
        term_inside_sqrt = 0 # 或者采取其他错误处理方式

    c_r = (1 - np.sqrt(term_inside_sqrt)) * 100

    return c_r

def evaluate_on_dataset(dataset_type: str, model: nn.Module, raw_data_path: str,
                        scaler: DataScaler, window_size: int, prediction_steps: int,
                        device: torch.device, batch_size: int):
    print(f"\nEvaluating on {dataset_type} set...")
    raw_data = load_and_clean_data(raw_data_path)
    if raw_data.empty:
        print(f"No data for {dataset_type} set. Skipping evaluation.")
        # Return structure: actual_values_all_steps, predicted_values_all_steps, mse_per_step, cr_per_step
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps

    scaled_data = scaler.transform(raw_data)
    X_eval, y_true_scaled_tensor = create_sequences(scaled_data, window_size, prediction_steps, device)

    if X_eval.nelement() == 0:
        print(f"Not enough data in {dataset_type} set to create sequences. Skipping evaluation.")
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps

    eval_dataset = TensorDataset(X_eval, y_true_scaled_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    y_pred_scaled = make_predictions(model, eval_loader, device) # Shape: (num_samples, PREDICTION_STEPS)

    if y_pred_scaled.size == 0:
        print(f"No predictions made for {dataset_type} set. Skipping evaluation.")
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps

    num_samples = y_pred_scaled.shape[0]
    actual_values_all_steps = np.zeros((num_samples, prediction_steps))
    predicted_values_all_steps = np.zeros((num_samples, prediction_steps))
    mse_per_step = []
    cr_per_step = []

    for i in range(prediction_steps):
        y_pred_scaled_step_i = y_pred_scaled[:, i]
        y_true_scaled_step_i = y_true_scaled_tensor[:, i].cpu().numpy()

        actual_values_step_i = scaler.inverse_transform(y_true_scaled_step_i.reshape(-1, 1)).flatten()
        predicted_values_step_i = scaler.inverse_transform(y_pred_scaled_step_i.reshape(-1, 1)).flatten()

        actual_values_all_steps[:, i] = actual_values_step_i
        predicted_values_all_steps[:, i] = predicted_values_step_i

        if np.isnan(actual_values_step_i).any() or np.isnan(predicted_values_step_i).any():
            print(f"Warning: NaNs found in actual or predicted values for {dataset_type} set, step {i+1} before MSE/CR calculation.")
            actual_values_step_i = np.nan_to_num(actual_values_step_i, nan=0.0)
            predicted_values_step_i = np.nan_to_num(predicted_values_step_i, nan=0.0)
            print("NaNs have been replaced with 0 for MSE/CR calculation for this step.")

        mse_step_i = mean_squared_error(actual_values_step_i, predicted_values_step_i)
        mse_per_step.append(mse_step_i)

        cr_accuracy_step_i = calculate_cr_accuracy(actual_values_step_i, predicted_values_step_i)
        cr_per_step.append(cr_accuracy_step_i)

        if i == 0: # Print for the first step as before
            print(f"{dataset_type} Set - Step 1 Prediction MSE: {mse_step_i:.4f}")
            print(f"{dataset_type} Set - Step 1 Prediction C_R: {cr_accuracy_step_i:.2f}%")

    return actual_values_all_steps, predicted_values_all_steps, mse_per_step, cr_per_step

# VI. 结果可视化模块
# 1. 损失曲线绘制函数
def plot_loss_curves(train_loss_history: list, val_loss_history: list, save_path: str,
                       network_params_str: str, learning_rate: float, actual_epochs: int):
    plt.figure(figsize=(12, 7))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Model Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    info_text = f"Network: {network_params_str}\nLR: {learning_rate}, Actual Epochs: {actual_epochs}"
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Loss curve saved to {save_path}")

# 2. 预测对比图绘制函数
def plot_predictions_comparison(actual_values: np.ndarray, predicted_values: np.ndarray,
                                  dataset_name: str, prediction_step_label: str, save_path: str,
                                  mse_value: float, cr_value: float, network_params_str: str, learning_rate: float, actual_epochs: int,
                                  max_points: int = 500):
    if actual_values.size == 0 or predicted_values.size == 0:
        print(f"No data to plot for {dataset_name} ({prediction_step_label}) predictions. Skipping plot.")
        return

    plt.figure(figsize=(17, 8))

    # actual_values and predicted_values are now 1D arrays for a specific step
    plot_actual = actual_values
    plot_predicted = predicted_values

    num_points_original = len(actual_values)

    if num_points_original > max_points:
        plot_actual = actual_values[-max_points:]
        plot_predicted = predicted_values[-max_points:]
        num_points_to_plot = max_points
    else:
        num_points_to_plot = num_points_original

    time_labels = []
    for i in range(num_points_to_plot):
        total_minutes = i * 15 # Assuming 15-minute intervals for x-axis labels
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        time_labels.append(f"{hours:02d}:{minutes:02d}")

    x_ticks_positions = range(num_points_to_plot)

    plt.plot(x_ticks_positions, plot_actual, label="Actual Power", color='blue', marker='.', linestyle='-')
    plt.plot(x_ticks_positions, plot_predicted, label=f"Predicted Power ({prediction_step_label})", color='red', linestyle='--')

    plt.title(f"{dataset_name} Set: Actual vs. Predicted Power ({prediction_step_label} Prediction)")

    tick_spacing = max(1, num_points_to_plot // 10 if num_points_to_plot > 0 else 1)

    actual_ticks_for_plot = [pos for i, pos in enumerate(x_ticks_positions) if i % tick_spacing == 0 and i < len(time_labels)]
    actual_labels_for_plot = [time_labels[i] for i, pos in enumerate(x_ticks_positions) if i % tick_spacing == 0 and i < len(time_labels)]

    if actual_ticks_for_plot:
        plt.xticks(ticks=actual_ticks_for_plot, labels=actual_labels_for_plot, rotation=45)

    plt.xlabel("Time (HH:MM relative to start of plotted segment)")
    plt.ylabel("Power")
    plt.legend(loc='upper right')
    plt.grid(True)

    info_text = (f"{dataset_name} ({prediction_step_label}) MSE: {mse_value:.4f}\n"
                 f"{dataset_name} ({prediction_step_label}) C_R: {cr_value:.2f}%\n"
                 f"Network: {network_params_str}\n"
                 f"LR: {learning_rate}, Actual Epochs: {actual_epochs}")
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect to make space for figtext
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Predictions comparison plot for {dataset_name} ({prediction_step_label}) saved to {save_path}")

# VII. 主执行流程
if __name__ == "__main__":
    print("--- Wind Power Prediction using Improved LSTM (CUDA GPU Version) ---")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: Training data file not found: {TRAIN_DATA_PATH}")
        print(f"Please ensure '{TRAIN_DATA_PATH}' is in the same directory as the script.")
        exit()
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: Test data file not found: {TEST_DATA_PATH}")
        print(f"Please ensure '{TEST_DATA_PATH}' is in the same directory as the script.")
        exit()

    data_scaler = DataScaler(method=NORMALIZATION_METHOD)

    print("\n--- Preparing DataLoaders ---")
    try:
        train_loader, val_loader = prepare_dataloaders(
            train_file=TRAIN_DATA_PATH,
            val_split_ratio=0.2,
            scaler=data_scaler,
            window_size=WINDOW_SIZE,
            prediction_steps=PREDICTION_STEPS,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    except ValueError as e:
        print(f"Error during DataLoader preparation: {e}")
        exit()

    if not train_loader.dataset or not val_loader.dataset:
        print("Failed to create non-empty DataLoaders. Exiting.")
        exit()

    print("\n--- Initializing Model ---")
    lstm_model = ImprovedLSTMModel(
        input_features=1,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout_rate=LSTM_DROPOUT,
        fc_layers_config=FC_LAYERS,
        output_steps=PREDICTION_STEPS
    ).to(DEVICE)

    # 使用加权损失函数
    criterion = WeightedMSELoss(prediction_steps=PREDICTION_STEPS, weight_decay=0.85)

    # 使用AdamW优化器，添加权重衰减
    optimizer = optim.AdamW(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 混合精度训练支持 (如果GPU支持) - 对于RTX 3050Ti可能不需要
    scaler = None
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

    # 只在显存充足时启用混合精度训练
    if torch.cuda.is_available() and gpu_memory_gb > 6.0 and torch.cuda.get_device_capability()[0] >= 7:
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("Mixed precision training enabled (AMP)")
        except ImportError:
            print("Mixed precision training not available")
    else:
        print("Mixed precision training disabled for 4GB GPU to save memory")

    # 编译模型以提高性能 (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            lstm_model = torch.compile(lstm_model)
            print("Model compiled for better performance")
    except Exception as e:
        print(f"Model compilation not available: {e}")

    # 显存使用检查
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理缓存
        initial_memory = torch.cuda.memory_allocated(DEVICE) / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f} GB")

        # 测试一个小批次以检查显存使用
        try:
            test_batch_size = min(BATCH_SIZE, 8)
            test_input = torch.randn(test_batch_size, WINDOW_SIZE, 1).to(DEVICE)
            with torch.no_grad():
                test_output = lstm_model(test_input)
            test_memory = torch.cuda.memory_allocated(DEVICE) / 1024**3
            print(f"Memory after model test: {test_memory:.2f} GB")
            del test_input, test_output
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  GPU memory insufficient for current settings!")
                print("Suggestion: Reduce BATCH_SIZE or LSTM_HIDDEN_SIZE")
                print("Continuing with current settings, but may encounter errors...")

    print("\n--- Starting Model Training ---")
    train_loss_hist = []
    val_loss_hist = []

    try:
        train_loss_hist, val_loss_hist = train_model(
            model=lstm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            max_epochs=MAX_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            device=DEVICE,
            model_save_path=MODEL_SAVE_PATH, # Uses CUDA specific model path
            scheduler=scheduler
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ GPU memory error during training!")
            print("💡 Try reducing these parameters:")
            print(f"   BATCH_SIZE: {BATCH_SIZE} → {BATCH_SIZE//2}")
            print(f"   LSTM_HIDDEN_SIZE: {LSTM_HIDDEN_SIZE} → {LSTM_HIDDEN_SIZE//2}")
            print("   Or use the CPU version: LSTM_cpu_multi_steps.py")
            exit(1)  # 退出程序而不是抛出异常
        else:
            raise e  # 其他错误继续抛出

    actual_epochs_trained = len(train_loss_hist) if train_loss_hist else 0
    network_params_str = (f"LSTM(H:{LSTM_HIDDEN_SIZE}, L:{LSTM_NUM_LAYERS}, D:{LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0}), "
                          f"FC:{FC_LAYERS if FC_LAYERS else 'None'}")

    # 只有在训练成功时才进行可视化和评估
    if actual_epochs_trained > 0:
        print("\n--- Visualizing Training ---")
        loss_curve_save_path = os.path.join(FIGURE_SAVE_DIR, f"loss_curve_{timestamp_str}.png")
        plot_loss_curves(train_loss_hist, val_loss_hist, loss_curve_save_path,
                           network_params_str, LEARNING_RATE, actual_epochs_trained)
    else:
        print("\n--- Training failed, skipping visualization ---")

    if actual_epochs_trained > 0:
        print("\n--- Loading Best Model for Evaluation ---")
        if os.path.exists(MODEL_SAVE_PATH): # Uses CUDA specific model path
            if torch.cuda.is_available():
                lstm_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            else:
                lstm_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
            print(f"Model loaded from {MODEL_SAVE_PATH}")
        else:
            print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Evaluation will use the last trained model state.")

        # Evaluate on Train set
        actual_train_all, predicted_train_all, mse_train_per_step, cr_train_per_step = evaluate_on_dataset(
            dataset_type="Train",
            model=lstm_model,
            raw_data_path=TRAIN_DATA_PATH,
            scaler=data_scaler,
            window_size=WINDOW_SIZE,
            prediction_steps=PREDICTION_STEPS,
            device=DEVICE,
            batch_size=BATCH_SIZE
        )
    else:
        print("\n--- Skipping evaluation due to training failure ---")
        print("--- Script Finished with Errors ---")
        exit(1)

    # Evaluate on Test set
    actual_test_all, predicted_test_all, mse_test_per_step, cr_test_per_step = evaluate_on_dataset(
        dataset_type="Test",
        model=lstm_model,
        raw_data_path=TEST_DATA_PATH,
        scaler=data_scaler,
        window_size=WINDOW_SIZE,
        prediction_steps=PREDICTION_STEPS,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )

    # Print C_R table for the test set
    if cr_test_per_step and not any(np.isnan(cr_test_per_step)):
        print("\n--------------------------------------------------")
        print("Test Set - C_R Accuracy for Each Prediction Step")
        print("--------------------------------------------------")
        print(f"{'Lead Time':<15} | {'C_R (%)':>7}")
        print("--------------------------------------------------")
        for i, cr_value in enumerate(cr_test_per_step):
            lead_time_minutes = (i + 1) * 15
            if lead_time_minutes % 60 == 0:
                lead_time_str = f"{lead_time_minutes // 60}h"
            else:
                lead_time_str = f"{lead_time_minutes}min"
            print(f"{lead_time_str:<15} | {cr_value:>7.2f}")
        print("--------------------------------------------------")
    else:
        print("\nC_R values for test set are not available or contain NaNs, skipping C_R table.")

    # Define specific prediction steps to plot
    # (index, label)
    prediction_points_to_plot = [
        (0, "15min"),  # 1st point
        (3, "1h"),    # 4th point
        (7, "2h"),    # 8th point
        (11, "3h"),   # 12th point
        (15, "4h")    # 16th point
    ]

    print("\n--- Generating Prediction Comparison Plots for Specific Steps ---")
    # Plot for Train set
    if actual_train_all.size > 0 and predicted_train_all.size > 0:
        for step_index, label in prediction_points_to_plot:
            if step_index < PREDICTION_STEPS: # Ensure index is within bounds
                actual_train_step = actual_train_all[:, step_index]
                predicted_train_step = predicted_train_all[:, step_index]
                mse_train_step = mse_train_per_step[step_index]
                cr_train_step = cr_train_per_step[step_index]

                train_pred_plot_path = os.path.join(
                    FIGURE_SAVE_DIR,
                    f"train_predictions_comparison_{label.replace(' ', '')}_{timestamp_str}.png"
                )
                plot_predictions_comparison(
                    actual_values=actual_train_step,
                    predicted_values=predicted_train_step,
                    dataset_name="Train",
                    prediction_step_label=label,
                    save_path=train_pred_plot_path,
                    mse_value=mse_train_step,
                    cr_value=cr_train_step,
                    network_params_str=network_params_str,
                    learning_rate=LEARNING_RATE,
                    actual_epochs=actual_epochs_trained
                )
            else:
                print(f"Warning: Step index {step_index} for label '{label}' is out of bounds for PREDICTION_STEPS={PREDICTION_STEPS}. Skipping plot.")

    # Plot for Test set
    if actual_test_all.size > 0 and predicted_test_all.size > 0:
        for step_index, label in prediction_points_to_plot:
            if step_index < PREDICTION_STEPS: # Ensure index is within bounds
                actual_test_step = actual_test_all[:, step_index]
                predicted_test_step = predicted_test_all[:, step_index]
                mse_test_step = mse_test_per_step[step_index]
                cr_test_step = cr_test_per_step[step_index]

                test_pred_plot_path = os.path.join(
                    FIGURE_SAVE_DIR,
                    f"test_predictions_comparison_{label.replace(' ', '')}_{timestamp_str}.png"
                )
                plot_predictions_comparison(
                    actual_values=actual_test_step,
                    predicted_values=predicted_test_step,
                    dataset_name="Test",
                    prediction_step_label=label,
                    save_path=test_pred_plot_path,
                    mse_value=mse_test_step,
                    cr_value=cr_test_step,
                    network_params_str=network_params_str,
                    learning_rate=LEARNING_RATE,
                    actual_epochs=actual_epochs_trained
                )
            else:
                print(f"Warning: Step index {step_index} for label '{label}' is out of bounds for PREDICTION_STEPS={PREDICTION_STEPS}. Skipping plot.")

    print("\n--- Script Finished ---")