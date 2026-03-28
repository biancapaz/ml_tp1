#Funciones para calcular métricas

import numpy as np

def mse(y_true, y_pred): 

    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    err = y_pred - y_true
    res = float(np.mean(err**2))
    
    return res

def mae(y_true, y_pred):

    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    err = y_pred - y_true
    res = float(np.mean(np.abs(err)))

    return res

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))