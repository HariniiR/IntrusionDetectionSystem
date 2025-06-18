from NeuralNetwork import NeuralNetwork
from DataPreProcessing import DataPreProcessing
import numpy as np
from src.IntrusionDetectionSystem.DataPreProcessing import DataPreProcessing


class Predict:
    def __init__(self):
        self.model = NeuralNetwork.load_model("TrainedModel.npz")
        self.data = DataPreProcessing("TestingData.csv")
    def predict(self):
        inputData, expectedOutput, _, _ = self.data.getData()
        output = self.model.feedForward(inputData)
        return output, np.round(output), expectedOutput
    @staticmethod
    def calculate_accuracy(y_pred, y_true):
        # Convert probabilities or outputs to class labels (if needed)
        # If your output is already 0 or 1, skip this
        y_pred_labels = (y_pred > 0.5).astype(int)

        correct_predictions = np.sum(y_pred_labels == y_true)
        total_predictions = y_true.shape[1]

        accuracy = correct_predictions / total_predictions
        return accuracy