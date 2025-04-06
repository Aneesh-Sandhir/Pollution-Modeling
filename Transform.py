# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:15:11 2025

@author: anees
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class Transform:

    def __init__(self, csvPath, targetColumnIndicies = [0], categoricalColumnIndicies = [4]):
        self.values = pd.read_csv(csvPath, header = 0, index_col = 0).values
        self.targetColumnIndicies = targetColumnIndicies
        self.categoricalColumnIndicies = categoricalColumnIndicies
        self.encodeCategoricalValues(self.categoricalColumnIndicies)
        self.castValues()
        self.values = self.seriesToSupervised(pd.DataFrame(self.values), 1, 1, targetColumnIndicies)
        self.scaleValues()

    def encodeCategoricalValues(self, categoricalColumnIndicies = [4]):
        self.encoder = LabelEncoder()
        for categoricalColumnIndex in categoricalColumnIndicies:
            self.values[:,categoricalColumnIndex] = self.encoder.fit_transform(self.values[:,categoricalColumnIndex])

    def castValues(self):
        self.values = self.values.astype('float32')

    def scaleValues(self):    
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.values = self.scaler.fit_transform(self.values)
        
    def seriesToSupervised(self, data, nInput = 1, nOut = 1, dropnan = True, targetColumnIndicies = [0]):
        if (type(data) is list):
            nVariables = 1
        else:
            nVariables = data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(nInput, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(nVariables)]

        for i in range(0, nOut):
            cols.append(df[targetColumnIndicies].shift(-i))
            if (i == 0):
                names += [('var%d(t)' % (j + 1)) for j in targetColumnIndicies]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in targetColumnIndicies]

        agg = pd.concat(cols, axis = 1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace = True)

        return agg
