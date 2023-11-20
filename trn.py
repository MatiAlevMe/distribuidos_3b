# MLP's Trainig 

import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Training by use miniBatch iSGD
def train_miniBatch():
    ...
    return()

# mlp's training 
def train_mlp(x,y,param):        
    W,V = nn.iniWs(x.shape[0],y.shape[0],int(param[3]),int(param[4]))                    
    for Iter in range(1,int(param[1])):        
        xe,ye = nn.randpermute(x,y)
        yh, h = nn.forward(xe,W,param[5])
        T = int(x.shape[1]/param[1])
        ...      
        # if ((Iter %20)== 0):
            # print('Iter={} Cost={:.5f}'.format(Iter,costo[-1]))    
    # return(W,Costo) 


# Beginning ...
def main():
    param = dpar.load_config()
    x,y = dpar.load_dtrain()   
    # W,costo = train_mlp(x,y,param) 
    train_mlp(x,y,param)
    # dpar.save_ws_costo(W,costo)
       
if __name__ == '__main__':   
	 main()

