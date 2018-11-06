import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

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


# Separates data into segments for testing and training
# PARAM: collection a DataFrame of values read from pandas.
# RETURN: training_x_values,
# RETURN: test_x_value,
# RETURN: training_y_value,
# RETURN: test_y_values
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


# Performs 10-Fold cross validation using the data points and training with KNeighbors classifier.
# PARAM: training_input, x input of values,
# PARAM: training_output, y input of values,
# RETURN: training_error, the error from training
# RETURN: cv_error, cross validation error
def k_fold_validation(training_input, training_output, k):
    training_error = []
    cv_error = []
    k_folds = 10
    model = KNeighborsClassifier(n_neighbors=k)
    number_of_fold_indexes = int(len(training_input) / k_folds)
    for i in range(k_folds):
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(training_input) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = training_input.drop(training_input.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = training_output.drop(training_output.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        training_set_x = training_input.drop(training_input.index[index_start:index_end])
        training_set_y = training_output.drop(training_output.index[index_start:index_end])
        model.fit(training_set_x, training_set_y)
        cv_incorrect = 0
        training_incorrect = 0
        training_predictions = model.predict(training_set_x)
        cv_predictions = model.predict(validation_x)

        for j in range(len(training_predictions)):
            if training_predictions[j] != training_set_y.tolist()[j]:
                training_incorrect += 1
        for j in range(len(cv_predictions)):
            if cv_predictions[j] != validation_y.tolist()[j]:
                cv_incorrect += 1
        training_error.append(training_incorrect/len(training_predictions))
        cv_error.append(cv_incorrect/len(cv_predictions))
    print(len(training_error), len(cv_error))
    return np.mean(training_error), np.mean(cv_error)


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

training_error = []
cv_error = []
for k in lab_k_values:
    te, cve = k_fold_validation(training_x, training_y, k)
    training_error.append(te)
    cv_error.append(cve)

plt.plot(lab_k_values, training_error, label="Training Error")
plt.plot(lab_k_values, cv_error, label="Cross Validation Error")
plt.title("Classification error values based on K values")
plt.ylabel("Error rate for 10 fold cross validation")
plt.xlabel("K Nearest neighbors")
plt.legend()
plt.show()
