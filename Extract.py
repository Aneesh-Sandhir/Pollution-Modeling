# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:23:43 2025

@author: anees
"""
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class Extract:
    
    def __init__(self, sourcePath, destinationPath = 'extract.csv'):
        self.sourcePath = sourcePath
        self.destinationPath = destinationPath
        self.dataset = pd.read_csv(sourcePath)
        self.setIndex()
        self.dropColumns()
        self.renameColumns()
        self.identifyNonNumericColums()
        self.fillNullValues()
        self.dropLeadingObservations()
        self.exportToCSV()
    
    def setIndex(self):
        self.dataset.index = pd.to_datetime(self.dataset[['year', 'month', 'day', 'hour']])
    
    def dropColumns(self):
        columnsToDrop = ['No', 'year', 'month', 'day', 'hour']
        self.dataset.drop(columnsToDrop, axis = 1, inplace = True)
    
    def renameColumns(self):
        newColumnNames = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        self.dataset.columns = newColumnNames
        
    def identifyNonNumericColums(self):
        self.nonNumericColumns = self.dataset.select_dtypes(exclude=['number']).columns
        self.nonNumericColumnIndicies = [list(self.dataset.columns).index(column) for column in self.nonNumericColumns]
        
    def fillNullValues(self):
        self.dataset.fillna(0, inplace = True)
        
    def dropLeadingObservations(self, n = 24):
        self.dataset = self.dataset[n:]
        
    def exportToCSV(self):
        self.dataset.to_csv(self.destinationPath)
        
    def exploratoryDataAnalysis(self):
        df = self.dataset
        df = df.drop(columns = self.nonNumericColumns)
        plt.figure(figsize = (10,20))
        for index, column in enumerate(df.columns):
            plt.subplot(len(df.columns), 1, index + 1)
            plt.plot(df[column])
            plt.title(column, y = 0.75, loc = 'right')
        plt.show()
        
        