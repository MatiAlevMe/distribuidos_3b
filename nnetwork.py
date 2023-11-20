# Neural Network: functions

import numpy  as np

# initialize weights
"""
PARAM
m = N° de caracteristicas (X.shape[0])
K = N° de salidas (Y.shape[0])
L1 = N° de neuronas de la capa oculta 1
L2 (opcional) = N° de neuronas de la capa oculta 2

RETURN
(1) W1 = Peso capa oculta #1, W2 = Peso capa de salida
(2) W1 = Peso capa oculta #1, W2 = Peso capa oculta #2, W3 = Peso capa de salida
"""
def iniWs(m,K,L1,L2 = 0):
    W1 = randW(L1,m)
    V1 = np.zeros((L1,m))
    if L2 == 0:
        W2 = randW(K,L1)
        V2 = np.zeros((K,L1))
        return [W1,W2],[V1,V2]
    else:
        W2 = randW(L2,L1)
        V2 = np.zeros((L2,L1))
        W3 = randW(K,L2)
        V3 = np.zeros((K,L2))
        return [W1,W2,W3],[V1,V2,V3]
# Rand values for W    
def randW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
# Random location for data
def randpermute(x,y):
    J = x.shape[1]
    J_random = np.random.permutation(J)

    x = x[:, J_random]
    y = y[:, J_random]

    return x,y
#Activation function
def act_functions(function_number, x):
    if function_number == 1:
        # Sigmoid
        return 1 / (1 + np.exp(-x))
    elif function_number == 2:
        # Tanh
        return np.tanh(x)
    elif function_number == 3:
        # ReLU (Rectified Linear Unit)
        return np.maximum(0, x)
    elif function_number == 4:
        # ELU (Exponential Linear Unit)
        alpha = 0.01  # You can adjust the alpha parameter as needed
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elif function_number == 5:
        # SELU (Scaled Exponential Linear Unit)
        alpha = 1.67326  # Alpha parameter for SELU
        scale = 1.0507  # Scale parameter for SELU
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError("Invalid function_number. Please choose a number between 1 and 5.")
#Feed-forward
"""
PARAM
x = x permutados de entrenamiento
W = Vector de matrices peso por capa
fa = Tipo de función de activación 
""" 
def forward(x,W,fa):      
    h1 = np.dot(W[0], x)
    h1 = act_functions(fa, h1)
    
    if len(W) > 2:
        h2 = np.dot(W[1], h1)
        h2 = act_functions(fa, h2)

        yh = np.dot(W[2], h2)
        return yh,[h1,h2]
    else:
        yh = np.dot(W[1], h1)
        return yh,[h1]
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

