# Fetch Assignment: predicting Univariate Time Series Data
Author :  Anudeep Kumar
# Project tree

 * [LSTM Class](./MyLSTM.py)
 * [LSTM Trainer](./trainlstm.py)
 * [LSTM Predictor](./predictlstm.py)
 * [Flask Web App](./app.py)
 * [IPYNB for LSTM and Random Forest](./rfr.ipynb)
* [Experiments](./experiments)
   * [ARIMA IPYNB](./experiments/ARIMA.ipynb)
   * [arima.py](./experiments/arima.py)
   * [Model Saving Random Forest Regreesor](./experiments/modelSave.py)
   * [Predicting Random Forest Regressor](./experiments/predictor.py)
   * [Random Forest Class](./experiments/rfr.py)
   * [Random Forest Validation](./experiments/rfrValidation.py)
   * [Random Forest Model Saved](./experiments/rfmodel.pkl)
 * [Static data folder for Flask App](./static)
 * [Template Folder for Flask App](./template)
 * [Docker File](./Dockerfile)
 * [README.md](./README.md)

This github repo contains the solution to assignment requested. Below is the tree representing how the files are split. For this assignment the following approaches were experimented with :
* LSTM
* Random Forest Regressor
* ARIMA
* GARCH model for volatility analysis

Finally I have used an LSTM model fitted on the data to predict daily Receipt_Count for the year 2022 or any number of days required after 2021. It is wrapped in the ```app.py``` Flask Web App which predicts _**total monthly data for the year 2022 only**_

# Docker 
Link to Docker : https://hub.docker.com/r/anudeepk17/fetchassignment

Pull the image 
```
docker pull anudeepk17/fetchassignment:0.0.1.RELEASE
```

Run the docker with 
```
docker run -d -p 8080:5000 anudeepk17/fetchassignment:0.0.1.RELEASE
```
The issue that could come is of port. My Flask app uses port 5000. You can change it to [PORT_ON_YOUR_BROWSER] like 8080 or 4000, if 5000 is already being used by your system. 
```
docker run -d -p [PORT_ON_YOUR_BROWSER]:5000 <Image ID>
```

# Running Flask App Directly
 You can also run the ```app.py``` using the command
 ```
 python3 app.py 
 ```
 ```requirements.txt``` consists of the versions required and I have developed on ```Python 3.9.6```. 

If running into the following error when running th above command in terminal
 ```
Address already in use
Port 5000 is in use by another program......
``` 
Go to the code app.py and change the value of ```port``` in the last line to start server in a different port if 5000 is being used in your system.
```
app.run(host="0.0.0.0",port=5000,debug=True)
```
The app should be now hosted on 
```
localhost:[your_port_number]
http://127.0.0.1:[your_port_number]
http://192.168.1.199:[your_port_number]
```


# Python File Usage
### LSTM indpendent python files
Download the repository as it is.
The  ```predictlstm.py``` loads the model which was saved as h5 file and then in the code you can change the ```number_of_days```  variable after end of 2021 you want to predict and it should predict and plot the data. This allows the user to predict _**any number of future prediction**_ notlimited to 2022. It imports ```MyLSTM.py``` 

The ```trainlstm.py``` is used for training your own model and saving it as .h5 file. It imports ```MyLSTM.py```

```MyLSTM.py``` is file containing various utility files and the Neural network written from scratch.

### IPYNB : ```rfr.ipynb```
This ipynb file consists of all the experiments and trials of different algorithms I have used to try to predict the data. 
It starts with Random Forest Regressor written from scratch and trained with 17 days as input for next day prediction in a supervised manner.

Then I used  GARCH  or The generalized autoregressive conditional heteroskedasticity process  to estimating the volatility of our data.

I then use that to find an upper and lower bound for the data.

Then I used the LSTM approach and trained it for different epochs and nuerons and found out 2000 epochs and 4 neurons to be optimum and quick.

# ```experiments``` Folder
This folder consists of ```ARIMA.ipynb``` which is the ipynb where I experimented with ARIMA Model. The details are marked-down in the ipynb cell-by-cell. ```arima.py``` is the py file written from scratch for ARIMA model.

This folder also consists of ```rfr.py, rfrValidation.py, predictor.py, modelSave.py``` which are concerned files for Random Forest Regressor. These consists of Random Forest Regressor written from scratch, the Validation file for producing plot. 


 