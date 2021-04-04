import pandas as pd

data = pd.read_csv("LD2011_2014.txt",
                   parse_dates=[0],
                   delimiter=";",
                   decimal=",")
data.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)


def cons_peak(df):
    peak_df = df.copy()
    peak_df['timestamp'] = pd.to_datetime(peak_df['timestamp'])
    peak_df = peak_df[peak_df['timestamp'].dt.year == 2014]
    peak_df = peak_df.resample('1M', on='timestamp').mean()
    return peak_df.idxmax().dt.month


after = cons_peak(df=data)
print(after)
print(type(after))
print(after.index)
