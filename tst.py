# MLP's tesi
import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Begin
def main():
	param = dpar.load_config()
	x,y = dpar.load_data()
	W = dpar.load_ws()
	A = nn.forward(x,W,param[5])
	cm,Fsc = nn.metricas(y,A[-1]) 		
	dpar.save_metric(cm,Fsc)
	print('Matriz de Confusión:')
	print(cm)
	print('Fsc-mean {:.5f}'.format(Fsc*100))
	

if __name__ == '__main__':   
	 main()

