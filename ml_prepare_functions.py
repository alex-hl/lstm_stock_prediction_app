import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# csv_file = 'HTZ.csv' 
# df = pd.read_csv(csv_file)

def f1(df):
  
  df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
  df.index = df['Date']
  
  data = df.sort_index(ascending=True, axis=0)
  new_data  = data[['Date','Close']]
  new_data.drop('Date', axis=1,inplace=True)
  data_set = new_data.values
  train_size = int(data_set.shape[0] * 0.8)
  train = data_set[0:train_size,:]
  valid = data_set[train_size:,:]
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(data_set)
  #using timesteps of 60  
  X_train, y_train = [], []
  for i in range(60,len(train)):
      X_train.append(scaled_data[i-60:i,0])
      y_train.append(scaled_data[i,0])
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
  
  return X_train, y_train, new_data, train, valid, scaler, train_size


def f2(new_data, valid, scaler):
  
  inputs = new_data[len(new_data) - len(valid) - 60:].values
  inputs = inputs.reshape(-1,1)
  inputs  = scaler.transform(inputs)
  X_test = []
  for i in range(60,inputs.shape[0]):
      X_test.append(inputs[i-60:i,0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  
  return X_test


def f3(y_test, new_data, train, valid, scaler, train_size):
  
  y_test = scaler.inverse_transform(y_test)

  train = new_data[:train_size]
  valid = new_data[train_size:]
  valid['Predictions'] = y_test

  return train, valid

