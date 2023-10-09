from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
from rfr import MyRandomForestRegressor
from rfrValidation import series_to_supervised
import numpy as np
import pickle
series = read_csv('data_daily.csv', header=0, index_col=0)
values = series.values
# transform the time series data into supervised learning Here we take last 17 days as the input for predicting next day. So n_in=17
train = series_to_supervised(values, n_in=17)
# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
model = MyRandomForestRegressor(n_estimators=5000)
model.fit(trainX, trainy)
pickle.dump(model, open('model.pkl','wb'))
