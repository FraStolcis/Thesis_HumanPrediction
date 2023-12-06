from typing import Any
import TimeSeriesPreparation as tps
from numpy import linspace, polyval, polyfit
from scipy.interpolate import interp1d
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
         self.obj = [None for _ in range(len(self.Configs['CONFIG_DATA']['obj']))]
         self.obj_names = [None for _ in range(len(self.Configs['CONFIG_DATA']['obj']))]
         for obj in range(len(self.Configs['CONFIG_DATA']['obj'])):
                self.obj_names[obj] = self.Configs['CONFIG_DATA']['obj'][obj]['name']
                self.obj[obj] = tps.array((self.Configs['CONFIG_DATA']['obj'][obj]['x'],self.Configs['CONFIG_DATA']['obj'][obj]['y'],self.Configs['CONFIG_DATA']['obj'][obj]['z']))


    def plot(self):
  
        data  = tps.DataPreparation()
        X,y = data.data_preparation_multistep_dist(self.batch_size,self.interest_feature, self.n_steps_in, self.n_steps_out,self.target_feature, self.obj, self.obj_names)
        x_input = X[self.starting_pt]
        x_input = x_input.reshape((1, self.n_steps_in, len(self.target_feature)))

        model = load_model('model_LSTM_mv_ms_distance_mObj.h5')  
        yhat = tps.array(model.predict(x_input, verbose=0))        
        y_real = tps.array([X[self.starting_pt + _][-1]  for _ in range(self.n_steps_out)])
        y_real = y_real.reshape(( self.n_steps_out,len(self.target_feature)))
        yhat = yhat.reshape((self.n_steps_out))

        time_points = linspace(0, 800,num=self.n_steps_out)
        
        # Polynomial interpolation
        degree = 7 # Degree of the polynomial
        coefficients = polyfit(time_points, yhat, degree)

        # Generate interpolated values
        interpolated_values = polyval(coefficients, time_points)

        # New time points for a smoother curve
        
        smoothed_values = polyval(coefficients, time_points)
        X = tps.array([X[_][-1]  for _ in range(16146)])
        X = X.reshape(( 16146,len(self.target_feature)))
        
        plt.figure(1)
        plt.plot(smoothed_values, label = "yhat")
        plt.plot(y_real, label = "y_real")
       # dataset = tps.array([X[ _][-1]  for _ in range(X.shape[0])])
        '''
        plt.figure(2)
        plt.plot(yhat, label = "yhat")
        plt.plot(y_real, label = "y_real")
        '''
      
        
        plt.legend()
        
        plt.show()

      
        s = 'morgy'


ciao = Prediction()
ciao.plot()