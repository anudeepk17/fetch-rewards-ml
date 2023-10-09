 #importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template
from pandas import read_csv
from numpy import asarray
from MyLSTM import *
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Set the Matplotlib backend to 'Agg'
import matplotlib.pyplot as plt
import os



app=Flask(__name__,template_folder='template')
lstm_model = keras.models.load_model('my_lstm_model.h5')

# Compile the loaded model with the same optimizer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

@app.route('/') # Homepage
def home():
    return render_template('page2.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    
    idx=request.form["selected_month"]
    idx=idx.split("_")
    idx[0]=int(idx[0])
    idx[1]=int(idx[1])
    series = read_csv('data_daily.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
 
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
    for i in range(idx[1]):
        row = mytrain_scaled[i,-1].flatten()
        # make a one-step prediction
        yhat = forecast_lstm(lstm_model, 1, row)
        nxtiter=np.append(row,yhat)
        mytrain_scaled=np.concatenate((mytrain_scaled,nxtiter.reshape(1,-1)),axis=0)
        yhat = invert_scale(myscaler, row, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat)
        #print('Day=%d, Predicted=%f' % (365+i+1, yhat))
        predictions.append(yhat)
        raw_values=np.append(raw_values,int(yhat))
    
    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_file = '/static/graph.png'
    my_file2 = '/static/fullgraph.png'

    plt.plot(raw_values)
    plt.title('Prediction Curve')
    plt.xlabel('Day number')
    plt.ylabel('Receipt Count')
    plt.grid(True)
    savpath=my_path+my_file
    plt.savefig(savpath)    
    plt.clf()
    plt.plot(raw_values[365+idx[0]:365+idx[1]])
    plt.title('Prediction Curve For {}'.format(idx[2]))
    plt.xlabel('Day number')
    plt.ylabel('Receipt Count')
    plt.grid(True)
    savpath2=my_path+my_file2
    plt.savefig(savpath2)   
    plt.clf()


    # Convert BytesIO object to base64 string

    prediction=raw_values[365+idx[0]:365+idx[1]].sum()




    return render_template('page2.html', prediction_text='Approximate number of the scanned receipts for {} : {}'.format(idx[2],prediction),graphurl = savpath) # rendering the predicted result

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)