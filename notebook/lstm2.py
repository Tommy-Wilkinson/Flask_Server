# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:31:26 2020

@author: gagoyal
"""


import keras
from keras.models import Sequential
import numpy as np
from sklearn import preprocessing
from keras.layers import Dense, Dropout, LSTM, LeakyReLU
from sklearn.externals import joblib
from sklearn.utils import shuffle

dataset = "C:/Users/gagoyal/Downloads/CSV_data_Tuesday/CSVmaster.csv"

import pandas as pd
import os

lahead = 20

data = pd.read_csv(dataset,header=0)





#data = data[ data[0]< 130 ]

X= data.iloc[:,1:-1].values

#min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
#X = min_max_scaler.fit_transform(X)
#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)

y = data.iloc[:,-1].values
features_set = []
labels = []

i=0
while i <= X.shape[0]-30:
    features_set.append(X[i:i+30, :])
    labels.append(y[i])
    i+=30
    

labelEncoder = preprocessing.LabelEncoder()
labels = labelEncoder.fit_transform(labels)    
oneHotEncoder = preprocessing.OneHotEncoder(categorical_features=[0])
labels = labels.reshape(-1, 1)
labels = oneHotEncoder.fit_transform(labels).toarray()




features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 6))      
features_set, labels = shuffle(features_set, labels, random_state=0)



model = Sequential()
model.add(LSTM(150, input_shape=(features_set.shape[1], 6),return_sequences=True,dropout=0.1))
model.add(LSTM(units=100, return_sequences=False,dropout=0.1))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])


train_history = model.fit(x=features_set, y=labels,batch_size=10,epochs=200,shuffle=True,validation_split=0.1)


joblib.dump(model, 'model_2.pkl', compress=9)
joblib.dump(labelEncoder,'lencoder.pkl',compress=9)
joblib.dump(oneHotEncoder,'ohencoder.pkl',compress=9)













