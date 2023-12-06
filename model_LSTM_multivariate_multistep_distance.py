import TimeSeriesPreparation as tsp
from keras.models import Sequential, save_model
from keras.layers import Dense,LSTM, Conv1D,MaxPooling1D,Flatten




class ModelTest():
    def __init__(self):
        self.data =  tsp.DataPreparation()
        self.Configs = tsp.json.load(open('Config_model_LSTM_XYZ.json', 'r'))
        self.batch_size = self.Configs['CONFIG_DATA']['batch_size']
        self.interest_feature = self.Configs['CONFIG_DATA']['interest_feature']
        self.n_steps_in = self.Configs['CONFIG_DATA']['n_steps_in']
        self.n_steps_out = self.Configs['CONFIG_DATA']['n_steps_out']
        self.target_feature = self.Configs['CONFIG_DATA']['target_feature']
        self.epochs = self.Configs['CONFIG_DATA']['epochs']



    def ModelGen(self):

        data =   tsp.DataPreparation()
        X,y = data.data_preparation_multistep_dist(self.batch_size,self.interest_feature,self.n_steps_in, self.n_steps_out,self.target_feature)
        X,y = tsp.array(X),tsp.array(y)
        
        X = X.reshape((X.shape[0], X.shape[1], len(self.target_feature)))
        y = y.reshape((y.shape[0], y.shape[1], len(self.target_feature)))
        n_output = y.shape[1]*y.shape[2]
        y = y.reshape((y.shape[0], n_output))

        # define model
        model = Sequential()
        model.add(LSTM(50,activation= 'relu',return_sequences = True, input_shape=(self.n_steps_in,len(self.target_feature))))
        model.add(LSTM(50,activation= 'relu'))
        
        model.add(Dense(n_output))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # save model
        save_model(model, 'model_LSTM_mv_ms_distance.h5')


start = ModelTest()
start.ModelGen()