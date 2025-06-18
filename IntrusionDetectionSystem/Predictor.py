from Predict import Predict
inference = Predict()
predictions, classes, expectedOutput = inference.predict()
print("Accuracy of the model: ",inference.calculate_accuracy(predictions, expectedOutput))
print("Output:")
print(predictions)
print("Predicted class (0 = not at risk, 1 = at risk):")
print(classes)
