# Neural Network: functions

import numpy  as np

# initialize weights
def iniWs(hidden_size, input_size, output_size):    
    np.random.seed(42)  # Fija la semilla para obtener resultados reproducibles.
    W1 = np.random.randn(hidden_size, input_size)  # Pesos de la capa de entrada a la capa oculta.
    W2 = np.random.randn(output_size, hidden_size)  # Pesos de la capa oculta a la capa de salida.
    V1 = np.zeros((hidden_size, input_size))  # Inicialización del momento para W1.
    V2 = np.zeros((output_size, hidden_size))  # Inicialización del momento para W2.
    return W1, W2, V1, V2  # Devuelve los pesos y los términos de momento.
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

