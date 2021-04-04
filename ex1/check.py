# PyTorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import random


def generate_data(frequency: str, num_of_samples: int, start_date: pd.Timestamp, end_date: pd.Timestamp):
    data_dates = pd.date_range(start_date, end_date, freq=frequency)
    # create sinusoidal time seris from start-end time range
    data = {
        i: [np.sin(random.randint(1, 10)) for _ in range(len(data_dates))]
        for i in range(num_of_samples)
    }

    data_df = pd.DataFrame(index=pd.DatetimeIndex(data_dates), data=data)
    covariate_df = pd.DataFrame(index=data_df.index)

    # Adding calendar features
    covariate_df["yearly_cycle"] = np.sin(2 * np.pi * covariate_df.index.dayofyear / 366)
    covariate_df["weekly_cycle"] = np.sin(2 * np.pi * covariate_df.index.dayofweek / 7)

    return data_df, covariate_df


class MQRNNLoadDataset(Dataset):
    """
    Dataset impelentation for MQ-RNN.
    For sake of simplicity I'm dropping from this point forwarding the covariate features.
    """

    def __init__(self, series_data: pd.DataFrame, horizon_size: int):
        self.series_data = series_data
        self.horizon_size = horizon_size

    def __len__(self):
        return self.series_data.shape[1]

    def __getitem__(self, idx):
        current_series = self.series_data.iloc[: -self.horizon_size, idx]
        future_multi_horizons = []

        for i in range(1, self.horizon_size + 1):
            series_horizon = self.series_data.iloc[i: self.series_data.shape[0] - self.horizon_size + i, idx].to_numpy()
            future_multi_horizons.append(series_horizon)

        current_series_tensor = torch.tensor(current_series)
        future_series_tensor = torch.tensor(np.transpose(future_multi_horizons))
        cur_series_tensor = torch.unsqueeze(current_series_tensor, dim=1)
        future_series_tensor = torch.unsqueeze(future_series_tensor, dim=1)

        return cur_series_tensor, future_series_tensor


ts_data, ts_covariate = generate_data('12h', 200, "01-01-2020", "01-01-2021")
ds = MQRNNLoadDataset(ts_data, 10)
print(ds[4])