# MLP's tesi
import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Begin
def main():			
	x,y    = dpar.load_data()
	W      = dpar.load_ws()
	zv     = nn.forward(xv,W)      		
	cm,Fsc = nn.metricas() 		
	dpar.save_metric()
	print(Fsc*100)
	print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))
	

if __name__ == '__main__':   
	 main()

