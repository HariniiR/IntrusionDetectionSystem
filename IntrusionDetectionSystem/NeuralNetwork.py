import numpy as np
class NeuralNetwork:
    def __init__(self):
        self.inputData=[]
        self.expectedOutput=[]
        self.numberOfSamples=None
        self.numberOfFeatures=None
        self.layers=[]
        self.numberOfLayers=None
        self.weights = []
        self.biases = []
        self.caches = {}

    def initializeValues(self,inputData, expectedOutput, numberOfSamples, numberOfFeatures):
        self.inputData = np.array(inputData)
        self.expectedOutput = np.array(expectedOutput)
        self.numberOfSamples = numberOfSamples
        self.numberOfFeatures = numberOfFeatures
        self.layers = [numberOfFeatures,32,16,8,1]
        self.numberOfLayers = len(self.layers)
        for i in range(1,len(self.layers)):
            weight = np.random.randn(self.layers[i], self.layers[i-1])*np.sqrt(2 / self.layers[i-1])
            bias = np.random.randn(self.layers[i], 1)
            self.weights.append(weight)
            self.biases.append(bias)
    def feedForward(self, inputData):
        self.caches = {"activation0": inputData}
        predictions = inputData
        for i in range(1, self.numberOfLayers):
            weights = self.weights[i-1]
            biases = self.biases[i-1]
            weightsSum = np.dot(weights, predictions) + biases
            self.caches["z" + str(i)] = weightsSum
            weightsSum = np.array(weightsSum, dtype=np.float64)

            if i != len(self.layers) - 1:
                predictions = np.maximum(0,weightsSum) #ReLU
            else:
                predictions = 1/(1+np.exp(-weightsSum)) #Sigmoid
            self.caches["activation" + str(i)] = predictions
        return predictions
    @staticmethod
    def sigmoid_derivative(z):
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)

    @staticmethod
    def calculateLoss(y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backPropagate(self, expectedOutput):
        grads = {}

        if expectedOutput.ndim == 1:
            expectedOutput = expectedOutput.reshape(-1, 1)

        A_final = self.caches[f"activation{self.numberOfLayers-1}"]
        m = self.numberOfSamples # batch size

        dA = (A_final - expectedOutput) / m  # dL/dA for MSE

        for i in reversed(range(1, self.numberOfLayers)):
            A_prev = self.caches[f"activation{i - 1}"]
            Z = self.caches[f"z{i}"]
            W = self.weights[i - 1]

            if i == len(self.layers) - 1:
                dZ = dA * self.sigmoid_derivative(Z)  # Output layer: sigmoid
            else:
                dZ = dA * (Z > 0).astype(float)       # Hidden layers: ReLU

            dW = np.array(np.dot(dZ, A_prev.T), dtype = np.float64)                # Shape: (out_dim, in_dim)
            db = np.array(np.sum(dZ, axis=1, keepdims=True),dtype = np.float64) # Shape: (out_dim, 1)

            dA = np.dot(W.T, dZ)                       # Propagate to previous layer

            grads[f"dW{i}"] = dW
            grads[f"db{i}"] = db

        return grads

    def save_model(self, model_path):
        model = {}
        total_weights = len(self.weights)
        for iterator in range(1, total_weights+1):
            model[f"W{iterator}"] = self.weights[iterator-1]
            model[f"b{iterator}"] = self.biases[iterator-1]
        np.savez(model_path, **model)


    @staticmethod

    def load_model(filename):
        data = np.load(filename)

        # Initialize a new model using the same architecture
        model = NeuralNetwork()

        total_layers = len(data.files) // 2  # assuming W1...Wn and b1...bn pairs

        # Load weights and biases
        weights = []
        biases = []

        for i in range(1, total_layers + 1):
            weights.append(data[f"W{i}"])
            biases.append(data[f"b{i}"])

        model.weights = weights
        model.biases = biases
        model.layers = [weights[0].shape[1]] + [w.shape[0] for w in weights]
        model.numberOfLayers = len(model.layers)
        return model

