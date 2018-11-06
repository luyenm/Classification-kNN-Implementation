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
training_set_percentage = 0.75
test_percentage = 1 - training_set_percentage

lab_k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]


# looks for K nearest neigbors
# PARAM: x, a test sample
# PARAM: Xtrain, an array of input vectors for the training set.
# PARAM: Ytrain, an array of output values for the training set.
# PARAM: k, the number of nearest neighbors.
# RETURN: y_pred, predicted value for the test sample.
def k_nearest_neighbors(x, xtrain, ytrain, k):
    results = [[]]

    # Performs euclidean distance calculations based on the test and training data, appends the y training data
    # to the array.
    for i in range(len(xtrain)):
        results.append([np.sqrt((x[0] - xtrain[i][0])**2 + (x[1] - xtrain[i][1]) ** 2), ytrain[i]])
    test_results = results[1:]
    test_results.sort()

    output_zero = 0
    output_one = 1

    # checks the y data and counts the outputs
    for i in range(k):
        if test_results[i][1] == 0:
            output_zero = output_zero + 1
        else:
            output_one = output_one + 1

    # compares if there are more 1's than 0's
    if output_one > output_zero:
        y_pred = 1
    else:
        y_pred = 0

    return y_pred


# Same function above, but using sklearn libraries.
def k_nearest_neighbors_function_calls(x, trainx, trainy, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainx, trainy)
    y_pred = model.predict([x])
    return y_pred


def separate_data(collection):
    indices = (len(collection))
    training_indices = int(indices * training_set_percentage)
    training_set = collection[:training_indices]
    test_set = collection[training_indices:]

    training_x_values = training_set.drop("digit", axis=1)
    test_x_values = test_set.drop("digit", axis=1)
    training_y_values = training_set["digit"]
    test_y_values = test_set["digit"]
    return training_x_values, test_x_values, training_y_values, test_y_values


print("Results from my implementation of kNN")
for i in range(len(part1_x)):
    test_x = part1_x[i]
    train_x = np.delete(part1_x, i, axis=0)
    test_y = part1_y[i]
    train_y = np.delete(part1_y, i, axis=0)
    print(k_nearest_neighbors(test_x, train_x, train_y, 3), part1_y[i])

print("Library calls for KNN:")
for i in range(len(part1_x)):
    test_x = part1_x[i]
    train_x = np.delete(part1_x, i, axis=0)
    test_y = part1_y[i]
    train_y = np.delete(part1_y, i, axis=0)
    print(k_nearest_neighbors_function_calls(test_x, train_x, train_y, 3), part1_y[i])

training_x, test_x, training_y, test_y = separate_data(data)
