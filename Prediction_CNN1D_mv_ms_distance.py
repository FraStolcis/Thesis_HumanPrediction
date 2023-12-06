from typing import Any
import TimeSeriesPreparation as tps

import matplotlib.pyplot as plt
from keras.models import load_model


class Prediction():

    def __init__(self):
         self.Configs = tps.json.load(open('Config_predictions.json', 'r'))
         self.batch_size = self.Configs['CONFIG_DATA']['batch_size']
         self.interest_feature = self.Configs['CONFIG_DATA']['interest_feature']
         self.target_feature = self.Configs['CONFIG_DATA']['target_feature']
         self.n_steps_in = self.Configs['CONFIG_DATA']['n_steps_in']
         self.starting_pt = self.Configs['CONFIG_DATA']['starting_pt']
         self.n_steps_out = self.Configs['CONFIG_DATA']['n_steps_out']  


    def plot(self):
  
        data  = tps.DataPreparation()
        X,y = data.data_preparation_multistep_dist(self.batch_size,self.interest_feature, self.n_steps_in, self.n_steps_out,self.target_feature)
        x_input = X[self.starting_pt]
        x_input = x_input.reshape((1, self.n_steps_in, len(self.target_feature)))

        model = load_model('model_LSTM_mv_ms_distance.h5')  
        yhat = tps.array(model.predict(x_input, verbose=0))
        '''

        for i in range(60):
           print("y", yhat[0][i+t])
           print("y", yhat[0][i+t + 1])
           print("x" , X[501 + i][-1]) 
           t = 1
        ''' 
        '''
        y_hat_z = [None for _ in range(int(self.n_steps_out))]
        t= 0
        
        for i in range(1,self.n_steps_out,2):
             y_hat_z[t] = yhat[0][i] 
             t = t + 1
        y_hat_z = tps.np.array(y_hat_z)     
        '''
        
        y_real = tps.array([X[self.starting_pt + _][-1]  for _ in range(self.n_steps_out)])
        y_real = y_real.reshape(( self.n_steps_out,len(self.target_feature)))
        yhat = yhat.reshape((self.n_steps_out,len(self.target_feature)))
        plt.plot(yhat, label = "yhat")
        plt.plot(y_real, label = "y_real_z")
        plt.legend()
        plt.show()
        s = 'morgy'


ciao = Prediction()
ciao.plot()