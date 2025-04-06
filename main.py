# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:33:00 2025

@author: anees
"""

from Extract import Extract
from Transform import Transform
from Load import Load

import math
import numpy as np
from sklearn.metrics import mean_squared_error

filePath = 'raw.csv'
extractObj = Extract(filePath)
print(extractObj.exploratoryDataAnalysis())

transformObj = Transform(extractObj.destinationPath, 
                         categoricalColumnIndicies = extractObj.nonNumericColumnIndicies)

loadObj = Load(transformObj.values)
print(loadObj.plotTrainingProgress())

def calculateRMSE(model, features, targets, scaler):
    
    yHat = model.predict(loadObj.testFeatures)
    features = features.reshape((features.shape[0], features.shape[2]))
    
    # invert scaling for actual
    invertedYHat = np.concatenate((yHat, features), axis = 1)
    invertedYHat = scaler.inverse_transform(invertedYHat)
    invertedYHat = invertedYHat[:,0]
    
    # invert scaling for actual
    targets = targets.reshape((len(targets), 1))
    invertedY = np.concatenate((targets, features), axis=1)
    invertedY = scaler.inverse_transform(invertedY)
    invertedY = invertedY[:,0]
    
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(invertedY, invertedYHat))
    return rmse

rmse = calculateRMSE(loadObj.model, loadObj.testFeatures, 
                     loadObj.testTargets, transformObj.scaler)
print('Test RMSE: %.3f' % rmse)

