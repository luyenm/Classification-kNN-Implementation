import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


data = pd.read_csv('data_lab7.csv')
part1_x = np.array([[1, 1],
                    [2, 3],
                    [3, 2],
                    [3, 4],
                    [2, 5]])
part1_y = [0, 0, 0, 1, 1]


# looks for K nearest neigbors
# PARAM: x, a test sample
# PARAM: Xtrain, an array of input vectors for the training set.
# PARAM: Ytrain, an array of output values for the training set.
# PARAM: k, the number of nearest neighbors.
# RETURN: y_pred, predicted value for the test sample.
def k_nearest_neightors(x, xtrain, ytrain, k):
    # model = KNeighborsClassifier(n_neighbors=k)
    # model.fit(xtrain, ytrain)
    # y_pred = model.predict([x])
    results = [[]]
    for i in range(len(xtrain)):
        results.append([np.sqrt((x[0] - xtrain[i][0])**2 + (x[1] - xtrain[i][1]) ** 2), ytrain[i]])
    test_results = results[1:]
    test_results.sort()

    output_zero = 0
    output_one = 1

    for i in range(k):
        if test_results[i][1] == 0:
            output_zero = output_zero + 1
        else:
            output_one = output_one + 1

    if output_one > output_zero:
        y_pred = 1
    else:
        y_pred = 0

    return y_pred


for i in range(len(part1_x)):
    test_x = part1_x[i]
    train_x = np.delete(part1_x, i, axis=0)
    test_y = part1_y[i]
    train_y = np.delete(part1_y, i, axis=0)
    print(k_nearest_neightors(test_x, train_x, train_y, 3), part1_y[i])
