#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Loading the dataset
#last 5 years data of APPLE Inc
train_dst = pd.read_csv("StockPred\AAPL.csv")


# In[3]:


train_dst.head()


# In[4]:


train_dst.shape


# In[5]:


train_dst.size


# In[6]:


train_dst.describe()


# In[7]:


train_dst.info()


# In[8]:


train_dst.isnull().sum()


# In[9]:


train_set = train_dst.iloc[:,1:2].values
print(train_set)
print(train_set.shape)


# In[10]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

t_scaler = MinMaxScaler(feature_range = (0,1)) 
scaler_train_set = t_scaler.fit_transform(train_set)
scaler_train_set


# In[11]:


#X and y train
X_train = []
Y_train = []
for i in range(60,1259):
    X_train.append(scaler_train_set[i-60:i, 0])
    Y_train.append(scaler_train_set[i, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)


# In[12]:


X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)


# In[13]:


#LSTM model

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[14]:


reg = Sequential()
reg.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
reg.add(Dropout(0.2))

reg.add(LSTM(units=50, return_sequences = True))
reg.add(Dropout(0.2))

reg.add(LSTM(units=50, return_sequences = True))
reg.add(Dropout(0.2))

reg.add(LSTM(units=50))
reg.add(Dropout(0.2))

reg.add(Dense(units=1))


# In[15]:


reg.compile(optimizer = 'adam', loss = 'mean_squared_error')
reg.fit(X_train, Y_train, epochs=100, batch_size=32)


# In[16]:


test_dst = pd.read_csv("StockPred\AAPL_latest.csv")
actual_stock = test_dst.iloc[:,1:2].values
print(test_dst.shape)
#print(actual_stock)

total_dst = pd.concat((train_dst['Open'], test_dst['Open']), axis = 0)
inputs = total_dst[len(total_dst)-len(test_dst)-60:].values

inputs = inputs.reshape(-1,1)
inputs = t_scaler.transform(inputs)

X_test = []
for i in range(60,300):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

print(X_test.shape)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1)) 


# In[17]:


predict_stock = reg.predict(X_test)
predict_stock = t_scaler.inverse_transform(predict_stock)


# In[18]:


plt.plot(actual_stock, color = 'blue', label = 'Actual Apple Stock Price')
plt.plot(predict_stock, color = 'green', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()


# In[ ]:




