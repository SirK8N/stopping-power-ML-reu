import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def plot_predictions_with_zoom(y_true, y_pred, y_col='target', offset=0,
                                zoom_range=(3000, 3200), log_scale=True):
    """
    Plots predicted vs. true values with MAE and a zoomed-in subplot.

    Parameters:
    - y_true: Array of ground truth values (will be shifted by offset).
    - y_pred: Array of predicted values.
    - y_col: Name of the target variable (for axis labeling).
    - offset: Integer offset if prediction starts after timesteps.
    - zoom_range: Tuple of (x_min, x_max) for the zoomed-in view.
    - log_scale: Whether to apply symlog scale to the y-axis.
    """
    if offset:
        y_true = y_true[offset:]
    
    mae = mean_absolute_error(y_true, y_pred)

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharey=True)

    # --- Full Plot ---
    axs[0].plot(y_true, label=f'True {y_col}', linewidth=2)
    axs[0].plot(y_pred, label=f'Predicted {y_col}', linewidth=2, linestyle='--')
    axs[0].set_title(f'Predicted vs. True {y_col.capitalize()} (Full)\nMAE = {mae:.4f} $E_h$', fontsize=14)
    axs[0].set_ylabel(f'{y_col.capitalize()}')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', linewidth=0.5)
    if log_scale:
        axs[0].set_yscale("symlog")

    # --- Zoomed-In Plot ---
    axs[1].plot(y_true, label=f'True {y_col}', linewidth=2)
    axs[1].plot(y_pred, label=f'Predicted {y_col}', linewidth=2, linestyle='--')
    axs[1].set_title(f'Zoomed In: Steps {zoom_range[0]}–{zoom_range[1]}', fontsize=13)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel(f'{y_col.capitalize()}')
    axs[1].set_xlim(zoom_range)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', linewidth=0.5)
    if log_scale:
        axs[1].set_yscale("symlog")

    plt.tight_layout()
    plt.show()


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

def prepare_sequences(
    data,
    X_cols,
    y_col,
    sequence_col='file_id',
    timestep_col='timestep',
    timesteps=5,
):
    """
    Extracts fixed-length sequences and corresponding targets from tabular time series data.

    Parameters:
    - data: pd.DataFrame, time series data containing features and a target.
    - X_cols: list of str, names of feature columns.
    - y_col: str, name of the target column.
    - sequence_col: str, name of the column to group sequences by (e.g., file ID).
    - timestep_col: str, name of the column to sort each group chronologically.
    - timesteps: int, number of timesteps per input sequence.

    Returns:
    - X: np.ndarray of shape (num_samples, timesteps, num_features), input sequences.
    - y: np.ndarray of shape (num_samples, 1), targets aligned with each sequence.
    - x_scaler: fitted StandardScaler for X_cols.
    - y_scaler: fitted StandardScaler for y_col.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Copy to avoid modifying original DataFrame
    data = data.copy()

    # Scale features and target
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    data[X_cols] = x_scaler.fit_transform(data[X_cols])
    data[y_col] = y_scaler.fit_transform(data[[y_col]])

    sequences, targets = [], []

    # Generate sequences grouped by `sequence_col`
    for _, group in data.groupby(sequence_col):
        group = group.sort_values(timestep_col)
        x_values = group[X_cols].values
        y_values = group[y_col].values

        for i in range(len(group) - timesteps):
            sequences.append(x_values[i:i + timesteps])
            targets.append(y_values[i + timesteps])

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets).reshape(-1, 1)

    return X, y, x_scaler, y_scaler

    
def create_datasets(X, y, batch_size=128, train_split=0.5):
    """
    Splits sequences into training and validation sets and wraps them in tf.data.Dataset objects.

    Parameters:
    - X: np.ndarray of shape (num_samples, timesteps, num_features), input sequences.
    - y: np.ndarray of shape (num_samples, 1), corresponding targets.
    - batch_size: int, number of samples per batch.
    - train_split: float in (0, 1), proportion of data to use for training.

    Returns:
    - train_ds: tf.data.Dataset, batched and shuffled training set.
    - val_ds: tf.data.Dataset, batched validation set.
    - input_shape: tuple, shape of a single input sample (timesteps, num_features).
    """
    import tensorflow as tf

    # Validate inputs
    num_samples = len(X)
    if num_samples != len(y):
        raise ValueError(f"X and y must have the same number of samples. Got {len(X)} and {len(y)}.")

    # Split into training and validation sets
    split_index = int(train_split * num_samples)
    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    # Wrap in tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Determine input shape
    input_shape = X.shape[1:]  # (timesteps, num_features)

    return train_ds, val_ds, input_shape


