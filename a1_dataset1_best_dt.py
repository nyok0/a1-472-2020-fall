
"""
========
START

Dataset 1
Best-DT: a better performing Decision Tree
found by performing grid search to find the best combination of hyper-parameters. For this, you need to experiment with the following parameter values:
• splitting criterion: gini and entropy
• maximum depth of the tree: 10 and no maximum
• minimum number of samples to split an internal node: experiment with values of your choice
• minimum impurity decrease: experiment with values of your choice
• class weight: None and balanced
"""
print(__doc__)

import matplotlib.pyplot as plt


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


from sklearn import tree

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

# Setup Tree Classifier
classifier = tree.DecisionTreeClassifier(
    # criterion='gini',
    criterion='entropy',
    max_depth=10,
    # max_depth=None,
    # min_samples_split=2,
    min_samples_split=5,
    # min_impurity_decrease=0.0,
    min_impurity_decrease=0.1,
    # class_weight=None
    class_weight="balanced"
)
classifierFit = classifier.fit(X_train, y_train)
predicted = classifierFit.predict(X_test)
# tree.plot_tree(classifierFit)

print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != predicted).sum()))

# print(predicted)

# Show Predictions Plot
showPrediction(dataTest, predicted, axes, nimages, letters, plt)

# REPORT + CONFUSION MATRIX
reportResults(classifier, X_test, y_test, predicted, "Best-DT-DS1.txt")

# Open plot windows
plt.show()

csvsave(predicted, "Best-DT-DS1.csv")