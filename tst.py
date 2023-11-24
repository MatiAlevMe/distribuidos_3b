# MLP's tesi
import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Predicción para múltiples clases (2 en este caso)
def procesar_lista(lista):
	resultado = []
	for sublista in lista:
		if sublista[0] > sublista[1]:
			resultado.append([1, 0])
		else:
			resultado.append([0, 1])
	return resultado

# Begin
def main():
	param = dpar.load_config()
	x,y = dpar.load_data()
	W = dpar.load_ws()
	A = nn.forward(x,W,param[5])
	
	y_pred = np.array(procesar_lista(A[-1].T)).T
	cm,Fsc = nn.metricas(y,y_pred) 		
	dpar.save_metric(cm,Fsc)
	print('Matriz de Confusión:')
	print(cm)
	print('Fsc-mean {:.5f}'.format(Fsc*100))
	

if __name__ == '__main__':   
	 main()

