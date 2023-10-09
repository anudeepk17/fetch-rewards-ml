from MyLSTM import *
from tensorflow import keras
import numpy as np
from pandas import read_csv

#Enter the number of days you want to predict after 2021
number_of_days=100

#Load the saved model from trainlstm.py
series = read_csv('data_daily.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
lstm_model = keras.models.load_model('my_lstm_model.h5') #Change this to the model you saved in trainlstm.py

# Compile the loaded model with the same optimizer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
 
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
 
# split data into train and test-sets
mytrain, mytest = supervised_values[:], supervised_values[:]
 
# transform the scale of the data
myscaler, mytrain_scaled, mytest_scaled = scale(mytrain, mytest)
predictions = list()
#The value in range() is the number of days later we want to predict from the last day of 2021
for i in range(number_of_days): 
    # We take the last value of our datset and send it to the network as input for prediction
    row = mytrain_scaled[i,-1].flatten() 
    # make a one-step prediction
    yhat = forecast_lstm(lstm_model, 1, row)
    # WE append the new prediction along with its input to the existing data so that next loop iteration we use that as an input
    nxtiter=np.append(row,yhat)
    mytrain_scaled=np.concatenate((mytrain_scaled,nxtiter.reshape(1,-1)),axis=0)
    yhat = invert_scale(myscaler, row, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat)
    print('Day=%d, Predicted=%f' % (365+i+1, yhat))
    predictions.append(yhat)
    raw_values=np.append(raw_values,int(yhat))
pyplot.plot(raw_values)
pyplot.show()



    
    
