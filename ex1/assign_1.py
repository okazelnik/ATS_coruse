import pandas as pd

data = pd.read_csv("LD2011_2014.txt",
                   parse_dates=[0],
                   delimiter=";",
                   decimal=",")
data.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
data = data.iloc[0:12]


def el_resample(df):
    resampled_df = df.resample('1H', on='timestamp').mean()
    resampled_df['timestamp'] = resampled_df.index
    resampled_df.reset_index(drop=True, inplace=True)
    return resampled_df


after = el_resample(df=data)
print(after.head())
print(after.tail())
print(set(data.columns.to_list()) == set(after.columns.to_list()))