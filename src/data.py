import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constants import FEATURE_COLUMNS, TARGET_COLUMN, RAW_DATA_PATH

def make_loaders(
        csv_path: str = RAW_DATA_PATH,
        batch_size: int = 16,
        test_size: float = 0.25,
        random_state: int = 42,
):
    """
    meta will contain the scaler stats + feature order.
    This is necessary for making inference match training.

    :param csv_path:
    :param batch_size:
    :param test_size:
    :param random_state:
    :return train_loader, val_loader, meta:
    """

    #1) Load my CSV
    df = pd.read_csv(str(csv_path))

    #2) Select features + target using our contract/constants files
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    #3) Split (train vs. val) so we can measure generalization,
    # maintain proportions for train and test sets with stratify

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    #4) Normalize numeric inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    #5) Convert to tensors for TensorDataset
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    #6) TensorDataset Stores my features and labels
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    #7) DataLoader batches + shuffling
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMN,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_version": "0.1.0",
        "threshold": 0.5
    }

    return train_loader, val_loader, meta