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
        raise ValueError("Invalid function_number. Please choose A number between 1 and 5.")
# dev Activation function
def der_act_functions(function_number, x):
    if function_number == 1:
        # Derivada de Sigmoid
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid)
    elif function_number == 2:
        # Derivada de Tanh
        tanh = np.tanh(x)
        return 1 - tanh**2
    elif function_number == 3:
        # Derivada de ReLU
        return np.where(x > 0, 1, 0)
    elif function_number == 4:
        # Derivada de ELU
        alpha = 0.01
        return np.where(x > 0, 1, alpha * np.exp(x))
    elif function_number == 5:
        # Derivada de SELU
        alpha = 1.67326
        scale = 1.0507
        return scale * np.where(x > 0, 1, alpha * np.exp(x))
    else:
        raise ValueError("Invalid function_number. Please choose A number between 1 and 5.")
#Feed-forward
"""
PARAM
x = x permutados de entrenamiento
W = Vector de matrices peso por capa
fa = Tipo de función de activación 
""" 
def forward(x,W,fa):
    A = []
    A.append(x)
    a = x
    for i in range(len(W)):
        a = np.dot(W[i],a)
        if i == len(W)-1:
           a = act_functions(1,a)
        else:
            a = act_functions(fa,a)
        A.append(a)
    return A
    
# Feed-Backward 
def gradWs(y,W,A,fa):
    # y = y.T
    gW = []

    delta = ((A[-1] - y)/y.shape[1]) * der_act_functions(1,A[-1])
    gW.insert(0, np.dot(A[-2],delta.T))

    for i in range(len(W) - 2, -1, -1):
        delta = np.dot(delta.T,W[i+1]).T*der_act_functions(fa,A[i+1])
        gW.insert(0, np.dot(A[i], delta.T))
    
    return gW
        
# Update MLP's weigth using iSGD
def updWs(beta,W,V,gW,mu):
    for i in range(len(W)):
        V[i] = beta * V[i] - mu * gW[i].T
        W[i] = W[i] + V[i]
    return W, V
def to_categorical(binary_output):
    return np.eye(binary_output.shape[0], dtype=int)[np.argmax(binary_output, axis=0)].T
# Measure
def metricas(y_true, y_pred):
    y_pred = to_categorical(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    P, R = calcular_precision_recall(cm)
    Fsc = calcular_fscore(P, R)
    return cm, Fsc

def calcular_precision_recall(cm):
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    P = TP / (TP + FP + 1e-10)
    R = TP / (TP + FN + 1e-10)
    return P, R

# Confusion matrix
def confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((y_true == i) & (y_pred == j))
    return cm

# Calculate F1 score
def calcular_fscore(P, R):
    fscore = 2 * (P * R) / (P + R + 1e-10)
    return fscore
#

