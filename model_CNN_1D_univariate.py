import TimeSeriesPreparation as tsp
from keras.models import Sequential
from keras.layers import Dense, Conv1D,MaxPooling1D,Flatten
import matplotlib.pyplot as plt


class ModelTest():

    def __init__(self):
        self.data =  tsp.DataPreparation()
        self.Configs = tsp.json.load(open('Config_model.json', 'r'))
        self.batch_size = self.Configs['CONFIG_DATA']['batch_size']
        self.interest_feature = self.Configs['CONFIG_DATA']['interest_feature']

    def ModelGen(self):

       
        X,y, n_steps,interest_feature = self.data.data_preparation(self.batch_size,self.interest_feature)
        X = X.reshape((X.shape[0], X.shape[1], interest_feature))
        # define model
        model = Sequential()
        model.add(Conv1D(64,60 , activation='relu', input_shape=(n_steps,interest_feature)))
        model.add(MaxPooling1D(pool_size=2, strides=2,padding='same'))
        model.add(Flatten())
        model.add(Dense(60, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=20, verbose=0)
        # demonstrate prediction
        x_input = X[600:660]
        x_input = x_input.reshape((60, n_steps, interest_feature))
        yhat = model.predict(x_input, verbose=0)
        
        print(yhat)
        for i in range(60):
           print(X[600 + i][-1]) 
         
        t = 4
      
       




ciao = ModelTest()
ciao.ModelGen()