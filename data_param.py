# Data and Parameters

import numpy  as np
#
def load_config():
    param = np.genfromtxt('config.csv', delimiter=',', dtype=float,encoding='utf-8')
    return param
# training data load
def load_dtrain():
    x = np.genfromtxt('xtrain.csv', delimiter=',',dtype=float,encoding='utf-8')
    y = np.genfromtxt('ytrain.csv', delimiter=',',dtype=float,encoding='utf-8')
    return x,y
#matrix of weights and costs
def save_ws_costo():
    ...
    return
#load pretrained weights
def load_ws():
    ...
    return
#
