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

# training data load
def load_data():
    x = np.genfromtxt('xtest.csv', delimiter=',',dtype=float,encoding='utf-8')
    y = np.genfromtxt('ytest.csv', delimiter=',',dtype=float,encoding='utf-8')
    return x,y
#matrix of weights and costs
def save_ws_costo(W,costo):
    np.savetxt("costo_avg.csv", costo, delimiter=',', fmt='%f')
    with open("pesos.csv", "w") as file:
        file.write(f"{len(W)}\n")
        for w in W:
            file.write(f"{len(w)}\n")
            for fila_w in w:
                fila = ','.join(map(str, fila_w))
                file.write(f"{fila}\n")
#load pretrained weights
def load_ws():
    pesos = []

    with open("pesos.csv", 'r') as file:
        num_pesos = int(file.readline().strip())

        for _ in range(num_pesos):
            num_filas = int(file.readline().strip())

            matriz_peso = []
            for _ in range(num_filas):
                fila = list(map(float, file.readline().strip().split(',')))
                matriz_peso.append(fila)

            pesos.append(np.array(matriz_peso))

    return pesos

def save_metric(cm,Fsc):
    # Escribir la matriz de confusión en cmatrix.csv
    np.savetxt('cmatrix.csv', cm, delimiter=',')

    # Escribir la puntuación F1 en fscores.csv
    np.savetxt('fscores.csv', [Fsc], delimiter=',')
