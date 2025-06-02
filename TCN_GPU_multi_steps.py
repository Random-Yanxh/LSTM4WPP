import torch
import torch.nn as nn
from torch.nn.utils import weight_norm # Added for TCN
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
from typing import Union, List
from datetime import datetime

# I. 项目设置与配置
# 1. 环境初始化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. 定义全局超参数
PREDICTION_STEPS = 16
WINDOW_SIZE = 96
# TCN Specific Hyperparameters - 优化训练速度
TCN_NUM_CHANNELS = [32, 64, 128] # 减少通道数以提高训练速度 (原来是 [64, 128, 256])
TCN_KERNEL_SIZE = 3 # Kernel size for TCN convolutions
TCN_DROPOUT = 0.2 # 减少dropout以提高训练速度

FC_LAYERS = [64, 32] # 减少FC层复杂度 (原来是 [128, 64, 32])
BATCH_SIZE = 64      # 增加批次大小以提高训练效率 (原来是 32)
LEARNING_RATE = 0.001 # 提高学习率以加快收敛 (原来是 0.0005)
MAX_EPOCHS = 50 # 减少最大训练轮数 (原来是 100)
EARLY_STOPPING_PATIENCE = 10 # 减少早停耐心值 (原来是 15)
EARLY_STOPPING_MIN_DELTA = 0.0001
NORMALIZATION_METHOD = "standard"
MODEL_SAVE_PATH = "./tcn_wpp_model_gpu.pth" # Changed model save path for TCN version
FIGURE_SAVE_DIR = "./figures_tcn_gpu" # Changed figure save dir for TCN version
TRAIN_DATA_PATH = "train_set.xlsx"
TEST_DATA_PATH = "test_set.xlsx"

# 3. 设置全局随机种子
SEED = 42
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(SEED)

# 确保图像保存目录存在
if not os.path.exists(FIGURE_SAVE_DIR):
    os.makedirs(FIGURE_SAVE_DIR)
    print(f"Created directory: {FIGURE_SAVE_DIR}")

# II. 数据处理模块 (保持不变)
# 1. 原始数据加载与清洗
def load_and_clean_data(file_path: str) -> pd.Series:
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
        if isinstance(data, pd.Series): data_np = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray): data_np = data.reshape(-1, 1) if data.ndim == 1 else data
        else: raise TypeError("Input data must be a Pandas Series or NumPy array.")
        if self.method == "minmax": self.scaler = MinMaxScaler()
        elif self.method == "standard": self.scaler = StandardScaler()
        self.scaler.fit(data_np)
        print("Scaler fitted.")

    def transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if self.scaler is None: raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        if isinstance(data, pd.Series): data_np = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray): data_np = data.reshape(-1, 1) if data.ndim == 1 else data
        else: raise TypeError("Input data must be a Pandas Series or NumPy array.")
        scaled_data = self.scaler.transform(data_np)
        print(f"Data transformed. Original shape: {data_np.shape}, Scaled shape: {scaled_data.shape}")
        return scaled_data.flatten()

    def fit_transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None: raise RuntimeError("Scaler has not been fitted yet.")
        if data.ndim == 1: data_np = data.reshape(-1, 1)
        elif data.ndim == 2 and data.shape[1] == 1: data_np = data
        elif data.ndim == 2 and data.shape[1] > 1 :
            print(f"Warning: inverse_transform called on multi-column data (shape {data.shape}). Assuming scaling on the first column or that all columns use the same scale.")
            data_np = data
        else: raise ValueError(f"Input data for inverse_transform has incompatible shape: {data.shape}")
        original_data = self.scaler.inverse_transform(data_np)
        print(f"Data inverse-transformed. Scaled shape: {data_np.shape}, Original shape: {original_data.shape}")
        return original_data

# 3. 序列数据构建
def create_sequences(data: np.ndarray, window_size: int, prediction_steps: int, device: torch.device):
    X_list, y_list = [], []
    if data.ndim == 1: data = data.reshape(-1, 1)
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

# III. TCN模型定义
class Chomp1d(nn.Module):
    """用于实现因果卷积的模块，移除卷积输出末尾的多余padding"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ImprovedTCNModel(nn.Module):
    def __init__(self, input_features: int, num_channels: List[int], kernel_size: int,
                 dropout_rate: float, fc_layers_config: list, output_steps: int):
        super(ImprovedTCNModel, self).__init__()
        self.output_steps = output_steps

        self.tcn = TemporalConvNet(num_inputs=input_features,
                                   num_channels=num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout_rate)

        self.dropout_fc = nn.Dropout(dropout_rate) # Dropout before FC layers

        # 全连接层
        fc_module_list = []
        # TCN的输出通道数是num_channels的最后一个元素
        current_dim = num_channels[-1]
        if fc_layers_config:
            for fc_hidden_dim in fc_layers_config:
                fc_module_list.append(nn.Linear(current_dim, fc_hidden_dim))
                fc_module_list.append(nn.BatchNorm1d(fc_hidden_dim)) # BatchNorm expects (N, C)
                fc_module_list.append(nn.ReLU())
                fc_module_list.append(nn.Dropout(dropout_rate)) # Dropout within FC block
                current_dim = fc_hidden_dim
        self.fc_layers = nn.Sequential(*fc_module_list)

        # 分步预测层
        self.step_predictors = nn.ModuleList([
            nn.Linear(current_dim, 1) for _ in range(output_steps)
        ])

        print("ImprovedTCNModel initialized.")
        print(f"  Input features: {input_features}")
        print(f"  TCN Num channels: {num_channels}, Kernel size: {kernel_size}, Dropout: {dropout_rate}")
        print(f"  FC Layers Config: {fc_layers_config}")
        print(f"  Output steps: {output_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_features)
        # TCN expects (batch_size, input_features, seq_len)
        x_tcn = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_tcn)  # Output shape: (batch_size, num_channels[-1], seq_len)

        # 使用TCN最后一个时间步的输出
        out = tcn_out[:, :, -1]
        out = self.dropout_fc(out)

        # 全连接层
        if hasattr(self, 'fc_layers') and len(self.fc_layers) > 0:
            out = self.fc_layers(out)

        # 分步预测
        predictions = []
        for i, predictor in enumerate(self.step_predictors):
            step_pred = predictor(out)
            predictions.append(step_pred)

        output = torch.cat(predictions, dim=1)  # (batch_size, output_steps)
        return output

# 改进的损失函数 (保持不变)
class WeightedMSELoss(nn.Module):
    def __init__(self, prediction_steps: int, weight_decay: float = 0.9):
        super(WeightedMSELoss, self).__init__()
        self.prediction_steps = prediction_steps
        self.weights = torch.tensor([weight_decay ** i for i in range(prediction_steps)])
        self.weights = self.weights / self.weights.sum()
        print(f"WeightedMSELoss initialized with weights: {self.weights}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(predictions.device)
        step_losses = torch.mean((predictions - targets) ** 2, dim=0)
        weighted_loss = torch.sum(step_losses * weights)
        return weighted_loss

# IV. 模型训练与验证模块 (基本保持不变, 修改模型初始化和参数传递)
# 1. 数据准备 (保持不变)
def prepare_dataloaders(train_file: str, val_split_ratio: float, scaler: DataScaler,
                        window_size: int, prediction_steps: int, batch_size: int,
                        device: torch.device):
    print("Preparing dataloaders...")
    raw_data = load_and_clean_data(train_file)
    if raw_data.empty: raise ValueError(f"No data loaded from {train_file}.")
    scaled_data = scaler.fit_transform(raw_data)
    split_index = int(len(scaled_data) * (1 - val_split_ratio))
    train_data_scaled, val_data_scaled = scaled_data[:split_index], scaled_data[split_index:]
    print(f"Train data scaled shape: {train_data_scaled.shape}, Validation data scaled shape: {val_data_scaled.shape}")
    if len(train_data_scaled) < window_size + prediction_steps or len(val_data_scaled) < window_size + prediction_steps:
        min_len = window_size + prediction_steps
        print(f"Warning: Train/Val set too small (min length {min_len}). Train: {len(train_data_scaled)}, Val: {len(val_data_scaled)}")
        empty_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)
        return empty_loader, empty_loader
    X_train, y_train = create_sequences(train_data_scaled, window_size, prediction_steps, device)
    X_val, y_val = create_sequences(val_data_scaled, window_size, prediction_steps, device)
    if X_train.nelement() == 0 or X_val.nelement() == 0:
        print("Warning: Sequence creation resulted in empty tensors.")
        empty_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)
        return empty_loader, empty_loader
    train_dataset, val_dataset = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train DataLoader: {len(train_loader)} batches, Val DataLoader: {len(val_loader)} batches.")
    return train_loader, val_loader

# 2. 训练主函数 (保持不变)
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                max_epochs: int, early_stopping_patience: int,
                early_stopping_min_delta: float, device: torch.device,
                model_save_path: str, scheduler=None):
    print("Starting model training...")
    train_loss_history, val_loss_history = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_temp_path = model_save_path.replace(".pth", "_temp.pth")

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
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
        print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s")

        if scheduler is not None: scheduler.step(avg_val_loss)

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
                model.load_state_dict(torch.load(best_model_temp_path, map_location=device))
            break
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")
    if os.path.exists(best_model_temp_path):
        os.remove(best_model_temp_path)
        print(f"Removed temporary best model file: {best_model_temp_path}")
    return train_loss_history, val_loss_history

# V. 模型评估与预测模块 (保持不变)
# 1. 通用预测函数
def make_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())
    if not all_predictions: return np.array([])
    return np.concatenate(all_predictions, axis=0)

# 2. 在指定数据集上进行评估
def calculate_cr_accuracy(P_M, P_P): # noqa: N802
    if not isinstance(P_M, np.ndarray): P_M = np.array(P_M)
    if not isinstance(P_P, np.ndarray): P_P = np.array(P_P)
    if P_M.shape != P_P.shape: raise ValueError(f"P_M shape: {P_M.shape}, P_P shape: {P_P.shape}")
    if P_M.size == 0: return 0.0
    N = len(P_M)
    R_values = np.zeros_like(P_M, dtype=float)
    mask_gt_02 = P_M > 0.2
    R_values[mask_gt_02] = (P_M[mask_gt_02] - P_P[mask_gt_02]) / P_M[mask_gt_02]
    mask_le_02 = P_M <= 0.2
    R_values[mask_le_02] = (P_M[mask_le_02] - P_P[mask_le_02]) / 0.2
    R_i_squared_sum = np.sum(R_values**2)
    if N == 0: return 0.0
    term_inside_sqrt = R_i_squared_sum / N
    if term_inside_sqrt < 0: term_inside_sqrt = 0
    c_r = (1 - np.sqrt(term_inside_sqrt)) * 100
    return c_r

def evaluate_on_dataset(dataset_type: str, model: nn.Module, raw_data_path: str,
                        scaler: DataScaler, window_size: int, prediction_steps: int,
                        device: torch.device, batch_size: int):
    print(f"\nEvaluating on {dataset_type} set...")
    raw_data = load_and_clean_data(raw_data_path)
    if raw_data.empty:
        print(f"No data for {dataset_type} set. Skipping evaluation.")
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps
    scaled_data = scaler.transform(raw_data)
    X_eval, y_true_scaled_tensor = create_sequences(scaled_data, window_size, prediction_steps, device)
    if X_eval.nelement() == 0:
        print(f"Not enough data in {dataset_type} set to create sequences. Skipping evaluation.")
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps
    eval_dataset = TensorDataset(X_eval, y_true_scaled_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    y_pred_scaled = make_predictions(model, eval_loader, device)
    if y_pred_scaled.size == 0:
        print(f"No predictions made for {dataset_type} set. Skipping evaluation.")
        return np.array([]), np.array([]), [np.nan] * prediction_steps, [np.nan] * prediction_steps
    num_samples = y_pred_scaled.shape[0]
    actual_values_all_steps = np.zeros((num_samples, prediction_steps))
    predicted_values_all_steps = np.zeros((num_samples, prediction_steps))
    mse_per_step, cr_per_step = [], []
    for i in range(prediction_steps):
        y_pred_scaled_step_i = y_pred_scaled[:, i]
        y_true_scaled_step_i = y_true_scaled_tensor[:, i].cpu().numpy()
        actual_values_step_i = scaler.inverse_transform(y_true_scaled_step_i.reshape(-1, 1)).flatten()
        predicted_values_step_i = scaler.inverse_transform(y_pred_scaled_step_i.reshape(-1, 1)).flatten()
        actual_values_all_steps[:, i] = actual_values_step_i
        predicted_values_all_steps[:, i] = predicted_values_step_i
        if np.isnan(actual_values_step_i).any() or np.isnan(predicted_values_step_i).any():
            print(f"Warning: NaNs in actual/predicted for {dataset_type}, step {i+1}. Replacing with 0.")
            actual_values_step_i = np.nan_to_num(actual_values_step_i, nan=0.0)
            predicted_values_step_i = np.nan_to_num(predicted_values_step_i, nan=0.0)
        mse_per_step.append(mean_squared_error(actual_values_step_i, predicted_values_step_i))
        cr_per_step.append(calculate_cr_accuracy(actual_values_step_i, predicted_values_step_i))
        if i == 0:
            print(f"{dataset_type} Set - Step 1 Prediction MSE: {mse_per_step[-1]:.4f}")
            print(f"{dataset_type} Set - Step 1 Prediction C_R: {cr_per_step[-1]:.2f}%")
    return actual_values_all_steps, predicted_values_all_steps, mse_per_step, cr_per_step

# VI. 结果可视化模块 (保持不变)
# 1. 损失曲线绘制函数
def plot_loss_curves(train_loss_history: list, val_loss_history: list, save_path: str,
                       network_params_str: str, learning_rate: float, actual_epochs: int):
    plt.figure(figsize=(12, 7))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Model Loss During Training"); plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    info_text = f"Network: {network_params_str}\nLR: {learning_rate}, Actual Epochs: {actual_epochs}"
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
    plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.savefig(save_path); plt.show(); plt.close()
    print(f"Loss curve saved to {save_path}")

# 2. 预测对比图绘制函数
def plot_predictions_comparison(actual_values: np.ndarray, predicted_values: np.ndarray,
                                  dataset_name: str, prediction_step_label: str, save_path: str,
                                  mse_value: float, cr_value: float, network_params_str: str, learning_rate: float, actual_epochs: int):
    if actual_values.size == 0 or predicted_values.size == 0:
        print(f"No data to plot for {dataset_name} ({prediction_step_label}). Skipping plot.")
        return

    plt.figure(figsize=(17, 8))

    # 绘制所有数据点，不进行截取
    plot_actual = actual_values
    plot_predicted = predicted_values
    num_points_to_plot = len(actual_values)

    # 使用从1开始的等差数列作为横轴
    x_axis = np.arange(1, num_points_to_plot + 1)

    plt.plot(x_axis, plot_actual, label="Actual Power", color='blue', marker='.', linestyle='-')
    plt.plot(x_axis, plot_predicted, label=f"Predicted Power ({prediction_step_label})", color='red', linestyle='--')

    plt.title(f"{dataset_name} Set: Actual vs. Predicted Power ({prediction_step_label} Prediction)")

    # 设置横轴标签，显示合理数量的刻度
    tick_spacing = max(1, num_points_to_plot // 10 if num_points_to_plot > 0 else 1)
    x_ticks = x_axis[::tick_spacing]
    plt.xticks(x_ticks)

    plt.xlabel("Data Point Index")
    plt.ylabel("Power")
    plt.legend(loc='upper right')
    plt.grid(True)

    info_text = (f"{dataset_name} ({prediction_step_label}) MSE: {mse_value:.4f}\n"
                 f"{dataset_name} ({prediction_step_label}) C_R: {cr_value:.2f}%\n"
                 f"Network: {network_params_str}\n"
                 f"LR: {learning_rate}, Actual Epochs: {actual_epochs}\n"
                 f"Total Points: {num_points_to_plot}")
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Predictions comparison plot for {dataset_name} ({prediction_step_label}) saved to {save_path}")

# 3. C_R准确率表格保存函数
def save_cr_accuracy_table(cr_per_step: list, save_path: str, dataset_name: str,
                          network_params_str: str, learning_rate: float, actual_epochs: int):
    """保存C_R准确率表格到文件"""
    if not cr_per_step or any(np.isnan(cr_per_step)):
        print(f"C_R values for {dataset_name.lower()} set are not available or contain NaNs, skipping file save.")
        return

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            # 写入文件头信息
            f.write(f"=== {dataset_name} Set - C_R Accuracy Report ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Network: {network_params_str}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Actual Epochs: {actual_epochs}\n")
            f.write(f"Total Prediction Steps: {len(cr_per_step)}\n\n")

            # 写入表格
            f.write("--------------------------------------------------\n")
            f.write(f"{dataset_name} Set - C_R Accuracy for Each Prediction Step\n")
            f.write("--------------------------------------------------\n")
            f.write(f"{'Lead Time':<15} | {'C_R (%)':>7}\n")
            f.write("--------------------------------------------------\n")

            for i, cr_value in enumerate(cr_per_step):
                lead_time_minutes = (i + 1) * 15
                if lead_time_minutes % 60 == 0:
                    lead_time_str = f"{lead_time_minutes // 60}h"
                else:
                    lead_time_str = f"{lead_time_minutes}min"
                f.write(f"{lead_time_str:<15} | {cr_value:>7.2f}\n")

            f.write("--------------------------------------------------\n")

            # 写入统计信息
            f.write(f"\nStatistics:\n")
            f.write(f"Average C_R: {np.mean(cr_per_step):.2f}%\n")
            f.write(f"Best C_R: {np.max(cr_per_step):.2f}% (at {((np.argmax(cr_per_step) + 1) * 15)}min)\n")
            f.write(f"Worst C_R: {np.min(cr_per_step):.2f}% (at {((np.argmin(cr_per_step) + 1) * 15)}min)\n")

        print(f"C_R accuracy table saved to {save_path}")

    except Exception as e:
        print(f"Error saving C_R accuracy table: {e}")

# VII. 主执行流程
if __name__ == "__main__":
    print("--- Wind Power Prediction using TCN (GPU Optimized Version) ---")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists(TRAIN_DATA_PATH): print(f"ERROR: Training data file not found: {TRAIN_DATA_PATH}"); exit()
    if not os.path.exists(TEST_DATA_PATH): print(f"ERROR: Test data file not found: {TEST_DATA_PATH}"); exit()

    data_scaler = DataScaler(method=NORMALIZATION_METHOD)

    print("\n--- Preparing DataLoaders ---")
    try:
        train_loader, val_loader = prepare_dataloaders(
            train_file=TRAIN_DATA_PATH, val_split_ratio=0.2, scaler=data_scaler,
            window_size=WINDOW_SIZE, prediction_steps=PREDICTION_STEPS,
            batch_size=BATCH_SIZE, device=DEVICE
        )
    except ValueError as e: print(f"Error during DataLoader preparation: {e}"); exit()

    if not train_loader.dataset or not val_loader.dataset: print("Failed to create non-empty DataLoaders. Exiting."); exit()

    print("\n--- Initializing Model ---")
    tcn_model = ImprovedTCNModel(
        input_features=1, # Univariate time series
        num_channels=TCN_NUM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout_rate=TCN_DROPOUT,
        fc_layers_config=FC_LAYERS,
        output_steps=PREDICTION_STEPS
    ).to(DEVICE)

    criterion = WeightedMSELoss(prediction_steps=PREDICTION_STEPS, weight_decay=0.85).to(DEVICE)
    optimizer = optim.AdamW(tcn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\n--- Starting Model Training ---")
    train_loss_hist, val_loss_hist = train_model(
        model=tcn_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        device=DEVICE, model_save_path=MODEL_SAVE_PATH, scheduler=scheduler
    )

    actual_epochs_trained = len(train_loss_hist)
    network_params_str = (f"TCN(Channels:{TCN_NUM_CHANNELS}, Kernel:{TCN_KERNEL_SIZE}, Dropout:{TCN_DROPOUT}), "
                          f"FC:{FC_LAYERS if FC_LAYERS else 'None'}")

    print("\n--- Visualizing Training ---")
    loss_curve_save_path = os.path.join(FIGURE_SAVE_DIR, f"loss_curve_{timestamp_str}.png")
    plot_loss_curves(train_loss_hist, val_loss_hist, loss_curve_save_path,
                       network_params_str, LEARNING_RATE, actual_epochs_trained)

    print("\n--- Loading Best Model for Evaluation ---")
    if os.path.exists(MODEL_SAVE_PATH):
        tcn_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_SAVE_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Evaluation will use the last trained model state.")

    actual_train_all, predicted_train_all, mse_train_per_step, cr_train_per_step = evaluate_on_dataset(
        dataset_type="Train", model=tcn_model, raw_data_path=TRAIN_DATA_PATH, scaler=data_scaler,
        window_size=WINDOW_SIZE, prediction_steps=PREDICTION_STEPS, device=DEVICE, batch_size=BATCH_SIZE
    )

    actual_test_all, predicted_test_all, mse_test_per_step, cr_test_per_step = evaluate_on_dataset(
        dataset_type="Test", model=tcn_model, raw_data_path=TEST_DATA_PATH, scaler=data_scaler,
        window_size=WINDOW_SIZE, prediction_steps=PREDICTION_STEPS, device=DEVICE, batch_size=BATCH_SIZE
    )

    if cr_test_per_step and not any(np.isnan(cr_test_per_step)):
        print("\n--------------------------------------------------")
        print("Test Set - C_R Accuracy for Each Prediction Step")
        print("--------------------------------------------------")
        print(f"{'Lead Time':<15} | {'C_R (%)':>7}")
        print("--------------------------------------------------")
        for i, cr_value in enumerate(cr_test_per_step):
            lead_time_minutes = (i + 1) * 15
            lead_time_str = f"{lead_time_minutes // 60}h" if lead_time_minutes % 60 == 0 else f"{lead_time_minutes}min"
            print(f"{lead_time_str:<15} | {cr_value:>7.2f}")
        print("--------------------------------------------------")

        # 保存C_R准确率表格到文件
        cr_table_save_path = os.path.join(FIGURE_SAVE_DIR, f"test_CR_{timestamp_str}.txt")
        save_cr_accuracy_table(
            cr_per_step=cr_test_per_step,
            save_path=cr_table_save_path,
            dataset_name="Test",
            network_params_str=network_params_str,
            learning_rate=LEARNING_RATE,
            actual_epochs=actual_epochs_trained
        )
    else:
        print("\nC_R values for test set are not available or contain NaNs, skipping C_R table.")

    prediction_points_to_plot = [(0, "15min"), (3, "1h"), (7, "2h"), (11, "3h"), (15, "4h")]

    print("\n--- Generating Prediction Comparison Plots for Specific Steps ---")
    for dataset_prefix, actual_all, predicted_all, mse_per_step_ds, cr_per_step_ds in [
        ("Train", actual_train_all, predicted_train_all, mse_train_per_step, cr_train_per_step),
        ("Test", actual_test_all, predicted_test_all, mse_test_per_step, cr_test_per_step)
    ]:
        if actual_all.size > 0 and predicted_all.size > 0:
            for step_index, label in prediction_points_to_plot:
                if step_index < PREDICTION_STEPS:
                    actual_step = actual_all[:, step_index]
                    predicted_step = predicted_all[:, step_index]
                    mse_step = mse_per_step_ds[step_index]
                    cr_step = cr_per_step_ds[step_index]
                    plot_path = os.path.join(FIGURE_SAVE_DIR, f"{dataset_prefix.lower()}_predictions_comparison_{label.replace(' ', '')}_{timestamp_str}.png")
                    plot_predictions_comparison(
                        actual_values=actual_step, predicted_values=predicted_step, dataset_name=dataset_prefix,
                        prediction_step_label=label, save_path=plot_path, mse_value=mse_step, cr_value=cr_step,
                        network_params_str=network_params_str, learning_rate=LEARNING_RATE, actual_epochs=actual_epochs_trained
                    )
                else:
                    print(f"Warning: Step index {step_index} for label '{label}' is out of bounds. Skipping plot.")

    print("\n--- Script Finished ---")