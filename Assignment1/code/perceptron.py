import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    URL_ = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header=None)

    # make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1] == 'Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype='float64')
    return data


def perceptron(data, num_iter):
    features = data[:, :-1]
    labels = data[:, -1]

    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1] + 1))

    misclassified_ = []

    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            print(x.transpose())
            y = np.dot(w, x.transpose())

            target = 1.0 if (y > 0) else 0.0

            delta = (label.item(0, 0) - target)

            if (delta):  # misclassified
                misclassified += 1
                w += (delta * x)

        misclassified_.append(misclassified)
    return (w, misclassified_)

data = load_data()
num_iter = 10
w, misclassified_ = perceptron(data, num_iter)
print(w)

# w = w + learning_rate * (expected - predicted) * x
#
#
# # Make a prediction with weights
# def predict(row, weights):
#     activation = weights[0]
#     for i in range(len(row)-1):
#         activation += weights[i + 1] * row[i]
#
#     return 1.0 if activation >= 0.0 else 0.0

