import numpy as np
# import pandas as pd

# Especifica la ruta del archivo dentro de la carpeta
archivo = 'xtest.csv'
archivo2= 'xtrain.csv'
archivo3 = 'ytrain.csv'
archivo4 = 'ytest.csv'

# Carga los datos desde el archivo CSV utilizando Pandas
# datos = pd.read_csv(archivo)
# datos2 = pd.read_csv(archivo2)
# datos3 = pd.read_csv(archivo3)
# datos4 = pd.read_csv(archivo4)

# Convierte los datos a un np.array utilizando NumPy
# array_resultante = np.array(datos)
# array_resultante2 = np.array(datos2)
# array_resultante3 = np.array(datos3)
# array_resultante4 = np.array(datos4)

# num_filas, num_columnas = array_resultante.shape

param = np.genfromtxt(archivo, delimiter=',', dtype=float,encoding='utf-8')
param2 = np.genfromtxt(archivo2, delimiter=',', dtype=float,encoding='utf-8')
param3 = np.genfromtxt(archivo3, delimiter=',', dtype=float,encoding='utf-8')
param4 = np.genfromtxt(archivo4, delimiter=',', dtype=float,encoding='utf-8')
print(param4.shape[1])

# xtest = 2000 columnas
# xtrain = 8000 columnas
# ytrain = 8000 columnas
# ytest = 2000 columnas

# Ahora `array_resultante` es un np.array que contiene los datos del archivo
#print(array_resultante)
