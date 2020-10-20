
"""
========
START

Dataset 1
Best-MLP: a better performing Multi-Layered Perceptron
found by performing grid search to find the
best combination of hyper-parameters. For this, you need to experiment with the following parameter
values:
• activation function: sigmoid, tanh, relu and identity
• 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers
with 10+10
• solver: Adam and stochastic gradient descent
"""
print(__doc__)

import matplotlib.pyplot as plt


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier

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

# Setup MLP Classifier
classifier = MLPClassifier(
    alpha=1e-5,
    activation='logistic', # sigmoid
    # activation='tanh',
    # activation='relu',
    # activation='identity',
    # hidden_layer_sizes=(30, ),
    # hidden_layer_sizes=(50, ),
    hidden_layer_sizes=(10, 10),
    # solver='sgd',
    solver='adam',
    random_state=1
)
classifierFit = classifier.fit(X_train, y_train)
predicted = classifierFit.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != predicted).sum()))

# print(predicted)

# Show Predictions Plot
showPrediction(dataTest, predicted, axes, nimages, letters, plt)

# REPORT + CONFUSION MATRIX
reportResults(classifier, X_test, y_test, predicted, "Best-MLP-DS1.txt")

# Open plot windows
plt.show()

csvsave(predicted, "Best-MLP-DS1.csv")