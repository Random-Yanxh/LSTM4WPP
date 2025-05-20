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
from datetime import datetime # Added for timestamp

# I. 项目设置与配置
# 1. 环境初始化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. 定义全局超参数
PREDICTION_STEPS = 16
WINDOW_SIZE = 96
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
FC_LAYERS = [64, 32]  # 空列表表示无额外FC层
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100 # 实际运行时可以调大
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0001
NORMALIZATION_METHOD = "minmax"  # "minmax" or "standard"
MODEL_SAVE_PATH = "./lstm_wpp_model.pth"
FIGURE_SAVE_DIR = "./figures"
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
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

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
    
    # 确保数据为数值类型
    data_series = pd.to_numeric(data_series, errors='coerce')
    
    # 处理因强制转换或原始数据产生的NaN
    if data_series.isnull().any():
        print("Warning: NaNs found after numeric conversion. Applying ffill and bfill.")
        data_series = data_series.ffill() #向前填充
        data_series = data_series.bfill() #向后填充

    # 如果仍然存在NaN（例如，整个列都是非数字或空的），则用0填充
    if data_series.isnull().any():
        print("Warning: NaNs still present after ffill/bfill. Filling remaining NaNs with 0.")
        data_series = data_series.fillna(0)
        if data_series.isnull().all(): # 如果整个序列都是NaN，填充后也全是0
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
        return scaled_data.flatten() # Return 1D array

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
        elif data.ndim == 2 and data.shape[1] > 1 : # For multi-step predictions
             # We assume inverse_transform is called on each step independently if needed,
             # or on the first step for evaluation.
             # For simplicity, if multi-dim, assume it's (samples, features) and we scale all.
             # This might need adjustment based on specific use case for multi-step.
             # For now, let's assume it's called on a single feature column.
            print(f"Warning: inverse_transform called on multi-column data (shape {data.shape}). Assuming scaling on the first column or that all columns use the same scale.")
            data_np = data
        else:
            raise ValueError(f"Input data for inverse_transform has incompatible shape: {data.shape}")

        original_data = self.scaler.inverse_transform(data_np)
        print(f"Data inverse-transformed. Scaled shape: {data_np.shape}, Original shape: {original_data.shape}")
        return original_data # Return array with original dimensions as input (after reshape)

# 3. 序列数据构建
def create_sequences(data: np.ndarray, window_size: int, prediction_steps: int, device: torch.device):
    """
    接收标准化后的1D NumPy 数组。
    将数据转换为适用于LSTM的 (X, y) PyTorch张量。
    X (输入) 张量形状: (样本数, window_size, 1)。
    y (目标) 张量形状: (样本数, prediction_steps)。
    确保张量被移至已配置的PyTorch设备。
    """
    X_list, y_list = [], []
    if data.ndim == 1:
        data = data.reshape(-1, 1) # Ensure data is 2D for LSTM input_features=1

    for i in range(len(data) - window_size - prediction_steps + 1):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size : i + window_size + prediction_steps].flatten()) # y should be (prediction_steps,)

    if not X_list: # Handle cases where data is too short
        print("Warning: Not enough data to create sequences with the given window_size and prediction_steps.")
        return torch.empty(0, window_size, 1, device=device), torch.empty(0, prediction_steps, device=device)

    X = np.array(X_list)
    y = np.array(y_list)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    print(f"Sequences created. X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    return X_tensor, y_tensor

# III. LSTM模型定义 (PyTorch nn.Module)
class LSTMModel(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, num_layers: int, 
                 dropout_rate: float, fc_layers_config: list, output_steps: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0) # Dropout only if num_layers > 1
        
        # Optional Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optional Fully Connected layers
        fc_module_list = []
        current_dim = hidden_size
        if fc_layers_config:
            for fc_hidden_dim in fc_layers_config:
                fc_module_list.append(nn.Linear(current_dim, fc_hidden_dim))
                fc_module_list.append(nn.ReLU())
                # fc_module_list.append(nn.Dropout(dropout_rate)) # Optional: dropout within FC layers
                current_dim = fc_hidden_dim
        self.fc_layers = nn.Sequential(*fc_module_list)
        
        # Final output layer
        self.output_layer = nn.Linear(current_dim, output_steps)
        
        print("LSTMModel initialized.")
        print(f"  Input features: {input_features}")
        print(f"  Hidden size: {hidden_size}, Num layers: {num_layers}, LSTM Dropout: {dropout_rate if num_layers > 1 else 0}")
        print(f"  FC Layers Config: {fc_layers_config}")
        print(f"  Output steps: {output_steps}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, window_size, input_features)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We use the output from the last time step of the last layer
        # lstm_out is (batch, seq_len, hidden_size * num_directions)
        # We want the last time step: lstm_out[:, -1, :]
        out = lstm_out[:, -1, :] 
        # Alternatively, use hn:
        # If num_layers > 1, hn is (num_layers * num_directions, batch, hidden_size)
        # out = hn[-1, :, :] # Output of the last layer

        out = self.dropout(out) # Apply dropout after LSTM
        
        if hasattr(self, 'fc_layers') and len(self.fc_layers) > 0:
            out = self.fc_layers(out)
            
        out = self.output_layer(out) # Final prediction
        # out shape: (batch_size, output_steps)
        return out

# IV. 模型训练与验证模块
# 1. 数据准备
def prepare_dataloaders(train_file: str, val_split_ratio: float, scaler: DataScaler, 
                        window_size: int, prediction_steps: int, batch_size: int, 
                        device: torch.device):
    """
    加载并清洗原始训练数据。
    使用传入的 scaler 实例对全部训练数据进行 fit_transform 标准化。
    按时间顺序将标准化后的训练数据划分为训练子集和验证子集。
    为训练子集和验证子集分别调用 create_sequences 构建PyTorch序列数据。
    创建对应的 DataLoader。
    返回训练 DataLoader、验证 DataLoader。
    """
    print("Preparing dataloaders...")
    raw_data = load_and_clean_data(train_file)
    if raw_data.empty:
        raise ValueError(f"No data loaded from {train_file}. Cannot prepare dataloaders.")

    scaled_data = scaler.fit_transform(raw_data) # Fit and transform on the entire training dataset
    
    # Split data
    split_index = int(len(scaled_data) * (1 - val_split_ratio))
    train_data_scaled = scaled_data[:split_index]
    val_data_scaled = scaled_data[split_index:]
    
    print(f"Train data scaled shape: {train_data_scaled.shape}")
    print(f"Validation data scaled shape: {val_data_scaled.shape}")

    if len(train_data_scaled) < window_size + prediction_steps or len(val_data_scaled) < window_size + prediction_steps:
        min_len = window_size + prediction_steps
        print(f"Warning: Train or validation set too small for sequence creation (min length {min_len}).")
        print(f"Train length: {len(train_data_scaled)}, Val length: {len(val_data_scaled)}")
        # Return empty DataLoaders or handle as error
        empty_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)
        return empty_loader, empty_loader


    X_train, y_train = create_sequences(train_data_scaled, window_size, prediction_steps, device)
    X_val, y_val = create_sequences(val_data_scaled, window_size, prediction_steps, device)

    if X_train.nelement() == 0 or X_val.nelement() == 0: # Check if tensors are empty
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
                model_save_path: str):
    print("Starting model training...")
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_temp_path = 'best_model_temp.pth' # Temporary path for best model during training

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
        
        # Validation phase
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
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_temp_path) # Save best model
            print(f"Validation loss improved. Saved best model to {best_model_temp_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            if os.path.exists(best_model_temp_path):
                print(f"Loading best model from {best_model_temp_path}")
                model.load_state_dict(torch.load(best_model_temp_path))
            break
            
    # Save the final model (either the last one or the best one if early stopping occurred)
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")
    
    # Clean up temporary best model file
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
        for X_batch, _ in data_loader: # We only need X for predictions
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())
    
    if not all_predictions:
        return np.array([]) # Return empty if no predictions

    return np.concatenate(all_predictions, axis=0) # Shape (total_samples, PREDICTION_STEPS)

# 2. 在指定数据集上进行评估
def evaluate_on_dataset(dataset_type: str, model: nn.Module, raw_data_path: str,
                        scaler: DataScaler, window_size: int, prediction_steps: int,
                        device: torch.device, batch_size: int):
    print(f"\nEvaluating on {dataset_type} set...")
    raw_data = load_and_clean_data(raw_data_path)
    if raw_data.empty:
        print(f"No data for {dataset_type} set. Skipping evaluation.")
        return np.array([]), np.array([]), np.nan # Return mse as NaN

    scaled_data = scaler.transform(raw_data) # Use fitted scaler to transform

    X_eval, y_true_scaled_tensor = create_sequences(scaled_data, window_size, prediction_steps, device)

    if X_eval.nelement() == 0:
        print(f"Not enough data in {dataset_type} set to create sequences. Skipping evaluation.")
        return np.array([]), np.array([]), np.nan

    eval_dataset = TensorDataset(X_eval, y_true_scaled_tensor) # y_true_scaled_tensor is needed for DataLoader structure
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    y_pred_scaled = make_predictions(model, eval_loader, device) # (total_samples, PREDICTION_STEPS)

    if y_pred_scaled.size == 0:
        print(f"No predictions made for {dataset_type} set. Skipping evaluation.")
        return np.array([]), np.array([]), np.nan

    # Extract first step (t+1) for evaluation and plotting
    y_pred_scaled_step1 = y_pred_scaled[:, 0]
    y_true_scaled_step1 = y_true_scaled_tensor[:, 0].cpu().numpy()

    # Inverse transform
    # Scaler expects 2D input (n_samples, n_features=1)
    actual_values_step1 = scaler.inverse_transform(y_true_scaled_step1.reshape(-1, 1)).flatten()
    predicted_values_step1 = scaler.inverse_transform(y_pred_scaled_step1.reshape(-1, 1)).flatten()

    # Check for NaNs before MSE calculation
    if np.isnan(actual_values_step1).any() or np.isnan(predicted_values_step1).any():
        print(f"Warning: NaNs found in actual or predicted values for {dataset_type} set before MSE calculation.")
        print(f"NaNs in actual_values_step1: {np.isnan(actual_values_step1).sum()}")
        print(f"NaNs in predicted_values_step1: {np.isnan(predicted_values_step1).sum()}")
        # Option: Replace NaNs with a placeholder (e.g., 0) to allow MSE calculation, or handle as error
        actual_values_step1 = np.nan_to_num(actual_values_step1, nan=0.0)
        predicted_values_step1 = np.nan_to_num(predicted_values_step1, nan=0.0)
        print("NaNs have been replaced with 0 for MSE calculation.")

    # Calculate MSE
    mse = mean_squared_error(actual_values_step1, predicted_values_step1)
    print(f"{dataset_type} Set - Step 1 Prediction MSE: {mse:.4f}")

    return actual_values_step1, predicted_values_step1, mse

# VI. 结果可视化模块
# 1. 损失曲线绘制函数
def plot_loss_curves(train_loss_history: list, val_loss_history: list, save_path: str,
                       network_params_str: str, learning_rate: float, actual_epochs: int):
    plt.figure(figsize=(12, 7)) # Increased figure size for text
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Model Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Add text information
    info_text = f"Network: {network_params_str}\nLR: {learning_rate}, Actual Epochs: {actual_epochs}"
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Loss curve saved to {save_path}")

# 2. 预测对比图绘制函数
def plot_predictions_comparison(actual_values: np.ndarray, predicted_values: np.ndarray,
                                dataset_name: str, save_path: str,
                                mse_value: float, network_params_str: str, learning_rate: float, actual_epochs: int,
                                max_points: int = 500):
    if actual_values.size == 0 or predicted_values.size == 0:
        print(f"No data to plot for {dataset_name} predictions. Skipping plot.")
        return

    plt.figure(figsize=(17, 8)) # Increased figure size for text
    
    plot_actual = actual_values
    plot_predicted = predicted_values
    
    num_points_original = len(actual_values)

    if num_points_original > max_points:
        plot_actual = actual_values[-max_points:]
        plot_predicted = predicted_values[-max_points:]
        num_points_to_plot = max_points
    else:
        num_points_to_plot = num_points_original
        
    # Generate time labels based on the number of points to be plotted
    time_labels = []
    for i in range(num_points_to_plot):
        total_minutes = i * 15
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        time_labels.append(f"{hours:02d}:{minutes:02d}")

    x_ticks_positions = range(num_points_to_plot)

    plt.plot(x_ticks_positions, plot_actual, label="Actual Power", color='blue', marker='.', linestyle='-')
    plt.plot(x_ticks_positions, plot_predicted, label="Predicted Power (Step 1)", color='red', linestyle='--')
    
    plt.title(f"{dataset_name} Set: Actual vs. Predicted Power (Step 1 Predictions)")
    
    # Set X-axis ticks and labels
    tick_spacing = max(1, num_points_to_plot // 10 if num_points_to_plot > 0 else 1)
    
    actual_ticks_for_plot = [pos for i, pos in enumerate(x_ticks_positions) if i % tick_spacing == 0 and i < len(time_labels)]
    actual_labels_for_plot = [time_labels[i] for i, pos in enumerate(x_ticks_positions) if i % tick_spacing == 0 and i < len(time_labels)]

    if actual_ticks_for_plot:
        plt.xticks(ticks=actual_ticks_for_plot, labels=actual_labels_for_plot, rotation=45)
    
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Power")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Add text information
    info_text = (f"{dataset_name} MSE: {mse_value:.4f}\n"
                 f"Network: {network_params_str}\n"
                 f"LR: {learning_rate}, Actual Epochs: {actual_epochs}")
    plt.figtext(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Predictions comparison plot for {dataset_name} saved to {save_path}")

# VII. 主执行流程
if __name__ == "__main__":
    print("--- Wind Power Prediction using LSTM ---")
    
    # Generate timestamp for filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 1. 执行全局配置和环境初始化 (already done at the top)
    # set_seed(SEED) is called at the top
    # FIGURE_SAVE_DIR is created at the top
    # DEVICE is printed at the top
    
    # Check if data files exist (using global constants)
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: Training data file not found: {TRAIN_DATA_PATH}")
        print(f"Please ensure '{TRAIN_DATA_PATH}' is in the same directory as the script.")
        exit()
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: Test data file not found: {TEST_DATA_PATH}")
        print(f"Please ensure '{TEST_DATA_PATH}' is in the same directory as the script.")
        exit()

    # 2. 实例化 DataScaler
    data_scaler = DataScaler(method=NORMALIZATION_METHOD)
    
    # 3. 训练模型
    print("\n--- Preparing DataLoaders ---")
    try:
        train_loader, val_loader = prepare_dataloaders(
            train_file=TRAIN_DATA_PATH,
            val_split_ratio=0.2, # Example split ratio
            scaler=data_scaler, # Scaler will be fitted here
            window_size=WINDOW_SIZE,
            prediction_steps=PREDICTION_STEPS,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    except ValueError as e:
        print(f"Error during DataLoader preparation: {e}")
        exit()

    if not train_loader.dataset or not val_loader.dataset: # Check if datasets are empty
        print("Failed to create non-empty DataLoaders. Exiting.")
        exit()

    print("\n--- Initializing Model ---")
    lstm_model = LSTMModel(
        input_features=1, # Single feature (power)
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout_rate=LSTM_DROPOUT,
        fc_layers_config=FC_LAYERS,
        output_steps=PREDICTION_STEPS
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Model Training ---")
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
        model_save_path=MODEL_SAVE_PATH
    )
    
    actual_epochs_trained = len(train_loss_hist)
    network_params_str = (f"LSTM(H:{LSTM_HIDDEN_SIZE}, L:{LSTM_NUM_LAYERS}, D:{LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0}), "
                          f"FC:{FC_LAYERS if FC_LAYERS else 'None'}")

    # 4. 可视化训练
    print("\n--- Visualizing Training ---")
    loss_curve_save_path = os.path.join(FIGURE_SAVE_DIR, f"loss_curve_{timestamp_str}.png")
    plot_loss_curves(train_loss_hist, val_loss_hist, loss_curve_save_path,
                       network_params_str, LEARNING_RATE, actual_epochs_trained)
    
    # 5. 加载最佳模型进行评估
    print("\n--- Loading Best Model for Evaluation ---")
    if os.path.exists(MODEL_SAVE_PATH):
        lstm_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_SAVE_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Evaluation will use the last trained model state.")

    # 6. 评估与可视化 (训练集)
    actual_train, predicted_train, mse_train = evaluate_on_dataset(
        dataset_type="Train",
        model=lstm_model,
        raw_data_path=TRAIN_DATA_PATH, # Evaluate on the whole training set
        scaler=data_scaler, # Use the already fitted scaler
        window_size=WINDOW_SIZE,
        prediction_steps=PREDICTION_STEPS,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    if actual_train.size > 0 and predicted_train.size > 0:
        train_pred_plot_path = os.path.join(FIGURE_SAVE_DIR, f"train_predictions_comparison_{timestamp_str}.png")
        plot_predictions_comparison(actual_train, predicted_train, "Train", train_pred_plot_path,
                                    mse_train, network_params_str, LEARNING_RATE, actual_epochs_trained)
    
    # 7. 评估与可视化 (测试集)
    actual_test, predicted_test, mse_test = evaluate_on_dataset(
        dataset_type="Test",
        model=lstm_model,
        raw_data_path=TEST_DATA_PATH,
        scaler=data_scaler, # Use the same scaler fitted on training data
        window_size=WINDOW_SIZE,
        prediction_steps=PREDICTION_STEPS,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    if actual_test.size > 0 and predicted_test.size > 0:
        test_pred_plot_path = os.path.join(FIGURE_SAVE_DIR, f"test_predictions_comparison_{timestamp_str}.png")
        plot_predictions_comparison(actual_test, predicted_test, "Test", test_pred_plot_path,
                                    mse_test, network_params_str, LEARNING_RATE, actual_epochs_trained)
        
    # 8. 打印最终的测试集MSE (已在 evaluate_on_dataset 中打印)
    print("\n--- Script Finished ---")