import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X
def compute_ln_norm_distance(vector1, vector2, n):
    """
    Arguments:
    vector1 -- A 1-dimensional array of size > 0.
    vector2 -- A 1-dimensional array of size equal to size of vector1
    n       -- n in Ln norm distance (>0)
    """
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
        abs_diff = abs(vector1[i] - vector2[i])
        diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance
def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    """
    Returns indices of 1st k - nearest neighbors in train_X, in order with nearest first.
    """
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
        distance = compute_ln_norm_distance(train_elem_x, test_example,n_in_ln_norm_distance)
        indices_dist_pairs.append([index, distance])
        index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices

def predict_target_values(test_X):
    train_X = np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.float64)
    n_in_ln_norm_distance = 2
    k = 2
    test_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n_in_ln_norm_distance)
        top_knn_labels = []
        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
        test_Y.append(most_frequent_label)
    return test_Y
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    pass 

    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close() 


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    pred_Y = np.asarray(pred_Y) 
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 