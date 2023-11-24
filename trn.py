# MLP's Trainig 

import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Training by use miniBatch iSGD
def train_miniBatch(x,y,param,W,V,beta):
    A = nn.forward(x,W,param[5])
    gW = nn.gradWs(y,W,A,param[5])
    W,V = nn.updWs(beta,W,V,gW,param[6])
    MSE = np.mean((A[-1] - y) ** 2)

    return W,V,MSE

# mlp's training 
def train_mlp(x,y,param):        
    W,V = nn.iniWs(x.shape[0],y.shape[0],int(param[3]),int(param[4]))
    T =  int(x.shape[1] / param[1])
    costo = []
    t = 0                    
    for Iter in range(1,int(param[0])+1):        
        xe,ye = nn.randpermute(x,y)

        for i in range(1,T+1):
            start = (i - 1) * int(param[1])
            end = i * int(param[1])
            x_batch = xe[:, start:end]
            y_batch = ye[:, start:end]

            tau = 1 - 1/T
            beta = (0.9*tau) / (0.1 + 0.9*tau)

            W,V,MSE = train_miniBatch(x_batch,y_batch,param, W,V,beta)
            costo.append(MSE)
        t += 1

        if ((Iter %20)== 0):
            print('Iter={} Cost={:.5f}'.format(Iter,costo[-1]))    
    return(W,costo)

# Beginning ...
def main():
    param = dpar.load_config()
    x,y = dpar.load_dtrain()   
    W,costo = train_mlp(x,y,param) 
    dpar.save_ws_costo(W,costo)
       
if __name__ == '__main__':   
	 main()

