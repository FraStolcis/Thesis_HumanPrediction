import TimeSeriesPreparation as tsp
from keras.models import Sequential, save_model
from keras.layers import Dense, LSTM,MaxPooling1D,Flatten,Dropout
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt



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
        self.obj = [None for _ in range(len(self.Configs['CONFIG_DATA']['obj']))]
        self.obj_names = [None for _ in range(len(self.Configs['CONFIG_DATA']['obj']))]
        for obj in range(len(self.Configs['CONFIG_DATA']['obj'])):
            self.obj_names[obj] = self.Configs['CONFIG_DATA']['obj'][obj]['name']    
            self.obj[obj] = tsp.array((self.Configs['CONFIG_DATA']['obj'][obj]['x'],self.Configs['CONFIG_DATA']['obj'][obj]['y'],self.Configs['CONFIG_DATA']['obj'][obj]['z']))



    def ModelGen(self):

        data =   tsp.DataPreparation()
        X,y = data.data_preparation_multistep_dist(self.batch_size,self.interest_feature,self.n_steps_in, self.n_steps_out,self.target_feature,self.obj,self.obj_names)
        X,y = tsp.array(X),tsp.array(y)
        
        X = X.reshape((X.shape[0], X.shape[1], len(self.target_feature)))
        y = y.reshape((y.shape[0], y.shape[1], len(self.target_feature)))
        n_output = y.shape[1]*y.shape[2]
        y = y.reshape((y.shape[0], n_output))
        X_train = X[:round(X.shape[0]*0.7),:,:]
        y_train = y[:round(y.shape[0]*0.7),:]
        X_val = X[round(X.shape[0]*0.7):,:,:]
        y_val = y[round(y.shape[0]*0.7):,:]
        '''
        X_train = tsp.array([X[ _][-1]  for _ in range(X.shape[0])])
        X_train = X_train.reshape(( X_train.shape[0],len(self.target_feature)))
        plt.plot(X_train)
        plt.show()
        '''
        # define model
        model = Sequential()
        model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(self.n_steps_in,len(self.target_feature))))
        model.add(LSTM(60, activation='relu'))
        model.add(Dense(n_output))
        cp = ModelCheckpoint('LSTM/', save_best_only=True) 

        model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])


        # fit model
        model.fit(X, y, epochs=self.epochs,  callbacks=[cp])
        # save model
        
        print(model.summary())
        
        save_model(model, 'model_LSTM_mv_ms_distance_mObj.h5')
        
        
        
      
       




ciao = ModelTest()
ciao.ModelGen()