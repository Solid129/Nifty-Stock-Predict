import re

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import LSTM, Dense,Activation
from keras.models import Sequential
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def series_to_new_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(df) is list else df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(' %s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(' %s(t)' % (df.columns[j])) for j in range(n_vars)]
        else:
            names += [(' %s(t+%d)' % (df.columns[j], i))for j in range(n_vars)]
        # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    else:
        agg = agg.fillna(0)
    return agg


def remove_columns_from_transformed_series(df2, main_column, negative, positive, current=False):
    columns_to_be_removed = []
    for i in range(0, df2.shape[1]):
        string1 = df2.columns[i]
        if re.search("\(t\-", string1):
            if not (re.search(main_column, string1)):
                if (negative == 1):
                    columns_to_be_removed.append(string1)
        if re.search("\(t\+", string1):
            if not (re.search(main_column, string1)):
                if (positive == 1):
                    columns_to_be_removed.append(string1)
        if "(t)" in string1:
            if not (re.search(main_column, string1)):
                if (current):
                    columns_to_be_removed.append(string1)
    df2.drop(columns_to_be_removed, axis=1, inplace=True)
    return df2

startYear = 2010
endYear = 2020

data = pd.read_csv('nifty_data{0}_{1}.csv'.format(startYear,endYear), sep='\t', index_col='Date')

#data = data.drop(columns=['P/B','Div Yield','Symbol','Expiry'])

timesteps = 5
t_features = 12
sequence = timesteps*t_features


# transforming data for LSTM model
data = series_to_new_supervised(data, timesteps, 1)

data = remove_columns_from_transformed_series(
    data, 'Close', 0, 1, current=True)


values = data.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)



# split into train and test sets
n_train = 1400
train = scaled_values[:n_train, :]
test = scaled_values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

""" print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print("Time Steps =", timesteps)
print("Number of Features considered =", t_features) """

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], timesteps, t_features))
test_X = test_X.reshape((test_X.shape[0], timesteps, t_features))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design LSTM network
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])

 # fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False) 


# make a prediction
inv_ypred = model.predict(test_X)
# inverse reshape of test_X array
test_X = test_X.reshape((test_X.shape[0], sequence))
#print (test_X.shape)


# invert scaling for forecast
inv_ypred = concatenate((test_X[:, :], inv_ypred), axis=1)
#print(inv_ypred.shape)
inv_ypred = scaler.inverse_transform(inv_ypred)
inv_ypred = inv_ypred[:,-1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#saving data to csv files
df = pd.DataFrame(list(zip(inv_y,inv_ypred)),index=data.index[n_train:],columns=['Nifty Actual','Nifty predicted'])
df.to_csv('predictedNifty{0}_{1}.csv'.format(startYear,endYear),sep='\t')

# Plot the Nifty with Actual v/s predicted
plt.figure(figsize=(15, 15))
plt.plot(inv_y, label='Nifty Actual')
plt.plot(inv_ypred, label='Nifty predicted')
plt.legend()
plt.show()
