import matplotlib.pyplot as plt

def plot_training_history(history, log_scale=True, style='fivethirtyeight'):
    """
    Plots training and validation loss & MAE from a Keras history object.

    Parameters:
    - history: Keras History object from model.fit()
    - log_scale: Whether to use symlog scale for y-axis (default: True)
    - style: Matplotlib style to use (default: 'fivethirtyeight')
    """
    #plt.style.use(style)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # --- Subplot 1: Loss ---
    axs[0].plot(history.history['loss'], label='Training Loss', color='tab:blue', linewidth=2)
    axs[0].plot(history.history['val_loss'], label='Validation Loss', color='tab:orange', linewidth=2)
    axs[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss (MSE)', fontsize=12)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()
    if log_scale:
        axs[0].set_yscale('symlog')
    axs[0].tick_params(labelsize=10)

    # --- Subplot 2: MAE ---
    axs[1].plot(history.history['mae'], label='Training MAE', color='tab:green', linewidth=2)
    axs[1].plot(history.history['val_mae'], label='Validation MAE', color='tab:red', linewidth=2)
    axs[1].set_title('Training vs Validation MAE', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('MAE', fontsize=12)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()
    if log_scale:
        axs[1].set_yscale('symlog')
    axs[1].tick_params(labelsize=10)

    plt.suptitle('Model Training Performance', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
def plot_loglog_mae(history, figsize=(3.5, 2.5), line_color='black'):
    plt.figure(figsize=figsize)
    plt.plot(history.history['mae'], label='Dense', color=line_color, linewidth=1.5)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel(r'MAE ($E_h / a_B$)', fontsize=10)

    plt.legend(frameon=False, loc='upper right')

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def prepare_sequence_datasets(
    data,
    X_cols,
    y_col,
    sequence_col='file_id',
    timestep_col='timestep',
    timesteps=5,
    batch_size=128,
    train_split=0.5,
):
    """
    Prepares train/val tf.data.Dataset from tabular time series data.

    Returns:
    - train_ds: tf.data.Dataset
    - val_ds: tf.data.Dataset
    - x_scaler: fitted StandardScaler for X
    - y_scaler: fitted StandardScaler for y
    - input_shape: tuple, shape of a single input sample (timesteps, num_features)
    """
    # Step 1: Scale features
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    data[X_cols] = x_scaler.fit_transform(data[X_cols])
    data[y_col] = y_scaler.fit_transform(data[[y_col]])

    sequences, targets = [], []

    # Step 2: Generate sequences
    for _, group in data.groupby(sequence_col):
        group = group.sort_values(timestep_col)
        x = group[X_cols].values
        y = group[y_col].values

        for i in range(len(group) - timesteps):
            sequences.append(x[i:i+timesteps])
            targets.append(y[i+timesteps])

    X = np.array(sequences)
    y = np.array(targets).reshape(-1, 1)

    # Step 3: Train/Validation Split
    num_samples = len(X)
    split_index = int(train_split * num_samples)

    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    # Step 4: Convert to tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Step 5: Derive input shape
    input_shape = (X.shape[1], X.shape[2])  # (timesteps, num_features)

    return train_ds, val_ds, x_scaler, y_scaler, input_shape

