# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:19:00 2025

@author: anees
"""

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib import pyplot as plt

class Load:
    
    def __init__(self, data, nTrainHours = 24 * 365, modelName = 'pollutionModel.keras', lossFunction = 'mae',
                 optimizer = 'adam', epochs = 50, batchSize = 72):
        self.values = data
        self.nTrainHours = nTrainHours
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        self.epochs = epochs
        self.batchSize = batchSize
        self.modelName = modelName
        self.splitData()
        self.reshapeData()
        self.trainModel()
    
    def splitData(self):
        self.train = self.values[:self.nTrainHours, :]
        self.test = self.values[self.nTrainHours:, :]
        self.trainFeatures, self.trainTargets = self.train[:, :-1], self.train[:, -1]
        self.testFeatures, self.testTargets = self.test[:, :-1], self.test[:, -1]

    def reshapeData(self):
        self.trainFeatures = self.trainFeatures.reshape((self.trainFeatures.shape[0], 1, self.trainFeatures.shape[1]))
        self.testFeatures = self.testFeatures.reshape((self.testFeatures.shape[0], 1, self.testFeatures.shape[1]))
    
    def trainModel(self):
        self.model = Sequential()
        self.model.add(Input(shape = (self.trainFeatures.shape[1], self.trainFeatures.shape[2]))) 
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss = self.lossFunction, optimizer = self.optimizer)
        self.history = self.model.fit(self.trainFeatures, self.trainTargets, epochs = self.epochs, 
                                      batch_size = self.batchSize, validation_data = (self.testFeatures, self.testTargets), 
                                      verbose = 2, shuffle = True)
        self.model.save(self.modelName)
        
    def plotTrainingProgress(self):
        plt.plot(self.history.history['loss'], label = 'train') 
        plt.plot(self.history.history['val_loss'], label = 'test') 
        plt.legend()
        plt.show
        
        