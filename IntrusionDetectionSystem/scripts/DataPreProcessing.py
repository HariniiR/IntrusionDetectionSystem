import pandas as pd
import numpy as np

class DataPreProcessing:
    def __init__(self, filename):
        self.filename = filename
        data = pd.read_csv(self.filename)

        # Drop rows with any non-numeric values in numeric columns (optional: for safety)
        self.inputData = data.drop(data.columns[-1], axis=1)
        self.expectedOutput = data.iloc[:, -1]

        # Convert labels: 'normal' → 0, all others → 1
        self.expectedOutput = (self.expectedOutput != 'normal').astype(int)

    @staticmethod
    def oneHotEncoding(inputData):
        return pd.get_dummies(inputData)

    @staticmethod
    def minMaxScaler(inputData):
        # Ensure all columns are numeric
        inputData = inputData.apply(pd.to_numeric, errors='coerce')
        minVal = inputData.min(axis=0)
        maxVal = inputData.max(axis=0)
        denominator = maxVal - minVal
        denominator[denominator == 0] = 1e-8
        return (inputData - minVal) / denominator

    def prepareData(self):
        # Handle categorical columns
        data = self.inputData[['protocol_type','service','flag']]
        dataEncoded = self.oneHotEncoding(data)

        # Reference: training encoding (ensures alignment)
        training = pd.read_csv("training_data.csv")
        trainingEncoded = self.oneHotEncoding(training[['protocol_type', 'service', 'flag']])
        dataEncoded = dataEncoded.reindex(columns=trainingEncoded.columns, fill_value=0)

        # Handle numerical columns
        numericColumns = self.inputData.drop(['protocol_type','service','flag'], axis=1)

        # Convert all columns to numeric, coercing errors
        numericColumns = numericColumns.apply(pd.to_numeric, errors='coerce')

        # Fill any NaNs (in case some string values snuck in)
        numericColumns = numericColumns.fillna(0)

        # Scale
        numericColumns = self.minMaxScaler(numericColumns)

        # Concatenate
        self.inputData = pd.concat([numericColumns, dataEncoded], axis=1)

    def getData(self):
        self.prepareData()
        numberOfSamples = self.inputData.shape[0]
        numberOfFeatures = self.inputData.shape[1]
        return self.inputData.T, self.expectedOutput.to_numpy().reshape(1, numberOfSamples), numberOfSamples, numberOfFeatures
