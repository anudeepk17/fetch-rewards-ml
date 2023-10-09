import pickle
from numpy import asarray
import numpy as np
from matplotlib import pyplot
filename='rfmodel.pkl'
model = pickle.load(open(filename, 'rb'))
n_in=17
# construct an input for a new prediction
for i in range(366):
    row = values[-n_in:].flatten()
    # make a one-step prediction
    yhat = model.predict(asarray([row]))
    print('Input: %s, Predicted: %.3f' % (row[-1], yhat[0]))
    values=np.append(values,[yhat[0]]).reshape(-1,1)

pyplot.plot(values, label='Predicted')
pyplot.legend()
pyplot.show()
