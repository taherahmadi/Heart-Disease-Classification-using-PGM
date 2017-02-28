import pandas as pd

# Reads train data from csv, returns pandas DF
def read(path):
    data = pd.read_csv(path, header=None, index_col=False)
    return data

# Reads train data from csv, returns pandas DF
def split_label(df, label_index):
    label = df.get(label_index)
    data = df.drop(label_index, axis=1)
    return data, label
