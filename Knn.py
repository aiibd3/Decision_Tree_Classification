import csv
import random
import numpy as np
from numpy import argsort


def calculate_std(array):
    mean = calculate_mean(array)
    std_sum = sum((x - mean) ** 2 for x in array)
    std = (std_sum / len(array)) ** 0.5
    return std


def calculate_mean(array):
    return sum(array) / len(array)


def normalize_features(features_train, features_test):
    features_train_normalized = np.zeros_like(features_train)
    features_test_normalized = np.zeros_like(features_test)
    for i in range(features_train.shape[1]):  # iterate over each feature column
        # Calculate mean and standard deviation for the feature column
        mean = calculate_mean(features_train[:, i])
        std = calculate_std(features_train[:, i])
        # Normalize training set feature column
        features_train_normalized[:, i] = (features_train[:, i] - mean) / std
        # Normalize test set feature column using mean and std of training set
        features_test_normalized[:, i] = (features_test[:, i] - mean) / std
    return features_train_normalized, features_test_normalized


class KNN:
    def _init_(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = []
        for x_train in self.X_train:
            dist = self._euclidean_distance(x, x_train)
            distances.append(dist)

        # Sort distances and return indices of the first k neighbors
        k_idx = argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]

        # Count the occurrences of each label
        label_counts = {}
        for label in k_neighbor_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Find the most common label
        most_common_labels = []
        max_count = max(label_counts.values())
        for label, count in label_counts.items():
            if count == max_count:
                most_common_labels.append(label)

        # If there is a tie, pick the label that comes first in the training data
        if len(most_common_labels) > 1:
            first_label_index = float('inf')
            for label in most_common_labels:
                index = self._find_label_index(k_neighbor_labels, label)
                if index < first_label_index:
                    most_common_label = label
                    first_label_index = index
        else:
            most_common_label = most_common_labels[0]

        return most_common_label

    def _euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5

    def _find_label_index(self, labels, label):
        for i in range(len(labels)):
            if labels[i] == label:
                return i
        return float('inf')


def accuracy(y_true, y_pred):
    correct_predictions = 0
    total_predictions = len(y_true)
    for i in range(total_predictions):
        if y_true[i] == y_pred[i]:
            correct_predictions += 1
    return correct_predictions, correct_predictions / total_predictions


def train_test_split_custom(X, y, test_ratio=0.3):
    # Calculate the number of samples for the testing set
    num_test = int(len(X) * test_ratio)

    # Shuffle the indices of the samples
    indices = list(range(len(X)))
    random.shuffle(indices)

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def load_data_from_csv(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            # Extract features (first three columns) as floats
            features = [float(value) for value in row[:3]]
            X.append(features)
            # Extract target label (last column) as integer
            label = int(row[-1])
            y.append(label)
    return X, y


def main():
    features, targets = load_data_from_csv('assets/BankNote_Authentication.csv')

    features_train, features_test, target_train, target_test = train_test_split_custom(features, targets, 0.3)

    features_train = np.array(features_train)
    features_test = np.array(features_test)
    features_train_normalized, features_test_normalized = normalize_features(features_train, features_test)
    print(features_train_normalized)
    print(features_test_normalized)

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for k_value in k_values:
        MY_KNN_MODEL = KNN(k_value)
        MY_KNN_MODEL.fit(features_train_normalized, target_train)
        prediction = MY_KNN_MODEL.predict(features_test_normalized)
        correct_prediction, accurcy = accuracy(target_test, prediction)
        total_instances = len(target_test)
        print("-------------------------------------------------------------------")
        print("K:", k_value)
        print("the number of correct predictions:", correct_prediction)
        print("the number of test instances:", total_instances)
        print("the accuracy:", accurcy)


if __name__ == "_main_":
    main()
