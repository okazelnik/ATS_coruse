from datetime import timedelta

import pandas as pd
import torch
import random

data = pd.read_csv("LD2011_2014.txt",
                   parse_dates=[0],
                   delimiter=";",
                   decimal=",")
data.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)


class ElDataset(torch.utils.data.Dataset):
    """Electricity dataset."""

    def __init__(self, df, samples):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per household.
        """
        self.raw_data = df.set_index("timestamp")
        self.raw_data = self.raw_data.resample('1H').mean()
        self.samples = samples
        self.min_date = None
        self.max_date = None
        self.mappings = None

        # setters
        self.set_min_max_dates()
        self.set_mappings()

    @staticmethod
    def create_household_from_id(household_id):
        if 0 <= household_id < 10:
            return f"MT_00{household_id}"
        elif 10 <= household_id <= 99:
            return f"MT_0{household_id}"
        else:
            return f"MT_{household_id}"

    def set_min_max_dates(self):
        self.min_date = min(self.raw_data.index).date()
        self.max_date = max(self.raw_data.index).date() - timedelta(days=8)

    def set_mappings(self):
        dates_pool = pd.date_range(start=self.min_date, end=self.max_date, freq='1H').tolist()
        self.mappings = {
            idx: pd.to_datetime(random.choice(dates_pool))
            for idx in range(self.__len__())
        }

    def __len__(self):
        return self.samples * self.raw_data.shape[1]

    def __getitem__(self, idx):
        household, start_training_ts = self.get_mapping(idx=idx)
        end_training_ts = pd.to_datetime(start_training_ts + timedelta(hours=167))  # takes 7 days for train
        start_test_ts = pd.to_datetime(end_training_ts + timedelta(hours=1))
        end_test_ts = pd.to_datetime(start_test_ts + timedelta(hours=23))  # takes 24 hours for test

        return self.raw_data.loc[start_training_ts:end_training_ts][household], \
               self.raw_data.loc[start_test_ts:end_test_ts][household]

        # return torch.tensor(self.raw_data.loc[start_training_ts:end_training_ts][household].values), \
        #    torch.tensor(self.raw_data.loc[start_test_ts:end_test_ts][household].values)

    def get_mapping(self, idx):
        household_id = (idx // self.samples) + 1
        return ElDataset.create_household_from_id(household_id), self.mappings[idx]


dataset = ElDataset(df=data, samples=2)
house, idxx = dataset.get_mapping(198)
print(idxx)
print(type(idxx))
print(house)
train, test = dataset[198]
print(train)
print(len(train))
print(test)
print(len(test))
