import torch
from ipynb.fs.full.interpolated_time_series import interpolated_time_series
# import ipynb.fs.full.sliding_window

def sliding_window(ts, features):
    X = []
    Y = []
    for i in range(features + 1, len(ts) + 1):
        X.append(ts[i - (features + 1): i - 1])
        Y.append([ts[i - 1]])
    
    return X, Y
    

def get_training_datasets(features, test_len):

    ts = interpolated_time_series()
    X, Y = sliding_window(ts, features)
    X_train, Y_train, X_test, Y_test = X[0:-test_len], Y[0:-test_len], X[-test_len:], Y[-test_len:]
    train_len = round(len(ts) * 0.7)
    X_train, X_val, Y_train, Y_val = X_train[0:train_len], X_train[train_len:], Y_train[0:train_len], Y_train[train_len:]
    x_train = torch.tensor(data = X_train)
    y_train = torch.tensor(data = Y_train)
    x_val = torch.tensor(data = X_val)
    y_val = torch.tensor(data = Y_val)
    x_test = torch.tensor(data = X_test)
    y_test = torch.tensor(data = Y_test)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == '__main__':
    ts = list(range(6))
    X, Y = sliding_window(ts, 40)
    print(f'Time series: {ts}')
    print(f'X: {X}')
    print(f'Y: {Y}')
    