
"""
========
START

Dataset 2
Base-MLP: a baseline Multi-Layered Perceptron
with 1 hidden layer of100 neurons, sigmoid/logistic
as activation function, stochastic gradient descent, and default values for the rest of the parameters.
"""
print(__doc__)

import matplotlib.pyplot as plt


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier

import numpy

from a1_functions import *




letters = [
    "pi","alpha","beta","sigma","gamma","delta","lambda","omega","mu","xi"
]


dataTrain, dataTrainResult = csvParser("dataset/train_2.csv")

# dataValidate, dataValidateResult = csvParser("dataset/val_2.csv")

# dataTest, dataTestResult = csvParser("dataset/val_2.csv")
dataTest, dataTestResult = csvParser("dataset/test_with_label_2.csv")


nimages = a1_init()

_, axes = plt.subplots(2, nimages, figsize=(nimages * 1.4, 7))

# Show Training Plot
showTraining(dataTrain, dataTrainResult, axes, nimages, letters, plt)


# DATA RESHAPE
n_samples = len(dataTrain)
data = dataTrain.reshape((n_samples, -1))

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split( data, dataTrainResult, test_size=0.25, shuffle=False  )

# SETUP TEST DATA
test_samples = len(dataTest)
testdata = dataTest.reshape((test_samples, -1))
X_test = testdata
y_test = dataTestResult

# Setup MLP Classifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifierFit = classifier.fit(X_train, y_train)
predicted = classifierFit.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != predicted).sum()))

# print(predicted)

# Show Predictions Plot
showPrediction(dataTest, predicted, axes, nimages, letters, plt)

# REPORT + CONFUSION MATRIX
reportResults(classifier, X_test, y_test, predicted, "Base-MLP-DS2.txt")

# Open plot windows
plt.show()

csvsave(predicted, "Base-MLP-DS2.csv")