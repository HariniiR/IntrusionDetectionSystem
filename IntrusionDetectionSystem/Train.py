from NeuralNetwork import NeuralNetwork
from src.IntrusionDetectionSystem.DataPreProcessing import DataPreProcessing
import numpy as np

class Train:
    def __init__(self):
        self.data = DataPreProcessing("TrainingData.csv")
        self.inputData, self.expectedOutput, self.numberOfSample, self.numberOfFeatures = self.data.getData()
        self.model = NeuralNetwork()
        self.model.initializeValues(self.inputData, self.expectedOutput, self.numberOfSample, self.numberOfFeatures)
        self.epochs = 100
        self.learning_rate = 0.5
    def train(self):
        for epoch in range(self.epochs):
            predictions = self.model.feedForward(self.inputData)
            #self.expectedOutput = np.array(self.expectedOutput).reshape(-1, 1)
            loss = self.model.calculateLoss(predictions,self.expectedOutput)
            grads = self.model.backPropagate(self.expectedOutput)

            for i in range(1, len(self.model.layers)):
                self.model.weights[i - 1] -= self.learning_rate * grads[f"dW{i}"]
                self.model.biases[i - 1] -= self.learning_rate * grads[f"db{i}"]

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Loss = {loss}")
        self.model.save_model("TrainedModel.npz")

obj = Train()
obj.train()