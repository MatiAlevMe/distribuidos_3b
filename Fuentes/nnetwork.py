# Neural Network: functions

import numpy  as np
    
# initialize weights
def iniWs():    
    ...
    return()
# Rand values for W    
def randW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
# Random location for data
def randpermute():
    ...
    return
#Activation function
def act_functions():
    ....
    return

#Feed-forward 
def forward():        
    ...
    return()
# Feed-Backward 
def gradWs():   
    ...    
    return()        
# Update MLP's weigth using iSGD
def updWs():
    ...        
    return(W)
# Measure
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    ...    
    return(cm,Fscore)
    
#Confusion matrix
def confusion_matrix():
    ...    
    return(cm)

#

