
"""
========
START

Dataset 1
PER: a Perceptron, with default parameter values.
"""
print(__doc__)

import matplotlib.pyplot as plt


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


from sklearn.linear_model import Perceptron

import numpy

from a1_functions import *




letters = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]


dataTrain, dataTrainResult = csvParser("dataset/train_1.csv")

# dataValidate, dataValidateResult = csvParser("dataset/val_1.csv")

# dataTest, dataTestResult = csvParser("dataset/val_1.csv")
dataTest, dataTestResult = csvParser("dataset/test_with_label_1.csv")


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

# Setup Perceptron Classifier
classifier = Perceptron(tol=1e-3, random_state=0)
classifierFit = classifier.fit(X_train, y_train)
predicted = classifierFit.predict(X_test)


print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != predicted).sum()))

# print(predicted)

# Show Predictions Plot
showPrediction(dataTest, predicted, axes, nimages, letters, plt)

# REPORT + CONFUSION MATRIX
reportResults(classifier, X_test, y_test, predicted, "PER-DS1.txt")

# Open plot windows
plt.show()

csvsave(predicted, "PER-DS1.csv")