from numpy import array, append, linalg
import pandas as pd
import json
import matplotlib.pyplot as plt
class DataPreparation():


    def __init__(self):

        self.Configs = json.load(open('config.json', 'r'))
        self.df  = [None for _ in range(len(self.Configs['CONFIG_DATA']['train_file_name']))]
        self.distance = [None for _ in range(len(self.Configs['CONFIG_DATA']['train_file_name']))]
        self.data = [None for _ in range(len(self.Configs['CONFIG_DATA']['train_file_name']))]
        self.df_test  = [None for _ in range(len(self.Configs['CONFIG_DATA']['test_file_name']))]
        self.n_features = self.Configs['CONFIG_DATA']['n_features']
        self.distance_mode = self.Configs['CONFIG_DATA']['distance_mode']
       
    
       
    
    def import_data(self,interest_feature):
       
       # importo data togliendo colonne inutili e rinominando le colonne    
        for i in range(len(self.Configs['CONFIG_DATA']['train_file_name'])):
            self.df[i] = pd.read_csv("INTENTION/" + self.Configs['CONFIG_DATA']['train_file_name'][i], sep=' ', header=None)
            self.df[i] = self.df[i].drop(columns=[0,1,2,3,7,8,9,10])
            self.df[i] = self.df[i].rename(columns={4: 'x', 5: 'y', 6: 'z'})
            self.df[i] = self.df[i][interest_feature ]
        '''    
        for i in range(len(self.Configs['CONFIG_DATA']['test_file_name'])):
            self.df_test[i] = pd.read_csv("INTENTION/" + self.Configs['CONFIG_DATA']['test_file_name'][i], sep=' ', header=None)
            self.df_test[i] = self.df_test[i].drop(columns=[0,1,2,3,7,8,9,10])
            self.df_test[i] = self.df_test[i].rename(columns={4: 'x', 5: 'y', 6: 'z'})
        '''     

    #calcolo distanza tra punto e oggetto   
    def dist_calc(self,distance,j,n_steps_out,obj):
        distance = array(distance)
        for i in range(len(self.df[j])):
            #controllo che ci siano le feature necessarie
            if self.n_features ==3:
                ee =   array((self.df[j]['x'][i] , self.df[j]['y'][i], self.df[j]['z'][i]))
                num_distance = linalg.norm(ee - obj)
                distance = append(distance, num_distance)
        for _ in range(n_steps_out):
           distance = append(distance,0.0) 


        return distance  # vettore distanza stessa lunghezza di data
    

    def data_preparation(self,batch_size,interest_feature,n_steps_in):   
        #IMPORTO DATA DA TABELLE IN FORMATO BUONO 
        self.import_data(interest_feature)
        #inizializzo sequence
        X, y = list(), list()

        
        #creo batch con tutti i dati di ogni file
        for i in range(batch_size):
           
            X, y = self.split_seq(X,y,i,n_steps_in )
        return array(X),array(y),n_steps_in, len(interest_feature )
    
    def data_preparation_multistep(self,batch_size,interest_feature,n_steps_in,n_steps_out): 
        #INPUT: quanti file, lista di feature di interesse, numero di step in input, numero di step in output  
        #OUTPUT: X e y in formato buono per il modello, numero di step in input, numero di feature di interesse

        self.import_data(interest_feature)
        X, y = list(), list()
        
        #ciclo per ogni file e appendo su X,y
        for i in range(batch_size):
            X, y = self.split_seq_multi(X,y,i,n_steps_in ,n_steps_out )
        return array(X),array(y),n_steps_in, len(interest_feature )
  
    
    def data_preparation_multistep_dist(self,batch_size,interest_feature,n_steps_in,n_steps_out,target_feature,obj):   
        #INPUT: quanti file, lista di feature di interesse, numero di step in input, numero di step in output  
        #OUTPUT: X e y in formato buono per il modello, numero di step in input, numero di feature di interesse

        self.import_data(interest_feature)
        X, y, distance = list(), list(), []
        
        #ciclo per ogni file e appendo su X,y
        for i in range(batch_size):
            distance = self.dist_calc(distance,i,n_steps_out,obj)
            X, y = self.split_seq_multi_dist(X,y,i,distance, n_steps_in ,n_steps_out,target_feature )
        return array(X),array(y)
    
   

    def split_seq(self,X,y,j,n_steps_in):
         if self.distance_mode == False:
       # credo batch di dati con grandezza n_steps
            for i in range(len(self.df[j])):
                end_ix = i + n_steps_in
                if end_ix > len(self.df[j])-1:
                    break
                seq_x, seq_y = self.df[j][i: end_ix], self.df[j].loc[end_ix]
                X = append(X,seq_x)
                y = append(y ,seq_y)
            return X, y
        
    def split_seq_multi(self,X,y,j, n_steps_in, n_steps_out):
       if self.distance_mode == False:
       # credo batch di dati con grandezza n_steps
            for i in range(len(self.df[j])):
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out 
                if out_end_ix > len(self.df[j]):
                    break
                seq_x, seq_y = self.df[j][i: end_ix], self.df[j][end_ix : out_end_ix]
                X = append(X,seq_x)
                y = append(y ,seq_y)
                return X, y
    
    def split_seq_multi_dist(self,X,y,j,distance, n_steps_in, n_steps_out,target_feature):
         
          #INPUT: X e y, indice del file, vettore distanza, numero di step in input, numero di step in output
          #OUTPUT: X e y in formato buono per il modello
        

          #ciclo per lunghezza distanza (stessa lunghezza di data[j]])
          for i in range((distance.shape[0])):
                
                #lunghezza X data da input
                end_ix = i + n_steps_in
                #lunghezza y data da fine input + output
                out_end_ix = end_ix + n_steps_out 
                # esco dal ciclo appena capisco che non ho più dati
                if out_end_ix > distance.shape[0]:
                 break
                #creo X e y 
                seq_x, seq_y = distance[i: end_ix], distance[end_ix : out_end_ix]
                X.append(seq_x)
                y.append(seq_y)
          return X, y
    
