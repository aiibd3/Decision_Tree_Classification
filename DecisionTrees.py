# Problem 1
#########################################################################
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def load_data_from_csv(file_path):
    features = []
    targets = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            # Extract features (first four columns) as floats
            feature_row = [float(value) for value in row[:4]]
            features.append(feature_row)
            # Extract target label (last column) as integer
            label = int(row[-1])
            targets.append(label)
    return features, targets


def run_experiment(features, targets, split_ratio, random_state=None):
    accuracies = []
    tree_sizes = []

    # Splitting the dataset into training and testing sets
    for _ in range(5):
        features_train, features_test, targets_train, targets_test = train_test_split(features, targets,
                                                                                      test_size=1 - split_ratio,
                                                                                      random_state=random_state)

        # Initializing the decision tree classifier
        classifier = DecisionTreeClassifier()  # Using entropy as the criterion
        # Training the classifier
        classifier.fit(features_train, targets_train)
        # Calculating accuracy on the testing set
        accuracy = classifier.score(features_test, targets_test)
        # Storing accuracy
        accuracies.append(accuracy)
        # Calculating the size of the decision tree
        tree_size = classifier.tree_.node_count
        # Storing the size of the decision tree
        tree_sizes.append(tree_size)

    return np.mean(accuracies), np.max(accuracies), np.min(accuracies), np.mean(tree_sizes), np.max(tree_sizes), np.min(
        tree_sizes)


def plot_results(split_ratios, results, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(split_ratios, [result for result in results], marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    file_path = 'assets/BankNote_Authentication.csv'
    features, targets = load_data_from_csv(file_path)

    print("Experiment 1: Fixed train_test split ratio (25%)")
    for i in range(5):
        mean_acc, max_acc, min_acc, mean_size, max_size, min_size = run_experiment(features, targets, split_ratio=0.25,
                                                                                   random_state=i)
        print(f"Experiment {i + 1}:")
        print(f"Accuracy - Mean: {mean_acc:.4f}, Max: {max_acc:.4f}, Min: {min_acc:.4f}")
        print(f"Tree Size - Mean: {mean_size}, Max: {max_size}, Min: {min_size}")
        print()

    # Experiment 2: Different range of train_test split ratio
    split_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    results_acc = []
    results_size = []

    print("\nExperiment 2: Different range of train_test split ratio")
    print(
        "Split Ratio    "
        "Mean Accuracy   "
        "Max Accuracy    "
        "Min Accuracy    "
        "Mean Tree Size   "
        "Max Tree Size     "
        "Min Tree Size")

    for split_ratio in split_ratios:
        accs = []
        sizes = []
        for random_state in range(5):
            mean_acc, max_acc, min_acc, mean_size, max_size, min_size = run_experiment(features, targets, split_ratio,
                                                                                       random_state=random_state)

            # print(split_ratio)
            accs.append(mean_acc)
            sizes.append(mean_size)
        results_acc.append(np.mean(accs))
        results_size.append(np.mean(sizes))

        # .0f is used to round the split ratio to 0 decimal places.
        # .4f is used to round the mean accuracy to 4 decimal places.
        print(
            f"{split_ratio * 100:.0f}-{(1 - split_ratio) * 100:.0f}             "
            f"{np.mean(accs):.4f}          "
            f"{np.max(accs):.4f}          "
            f"{np.min(accs):.4f}            "
            f"{np.mean(sizes)}           "
            f"{np.max(sizes)}              "
            f"{np.min(sizes)}")

    # Plot mean accuracy and mean number of nodes in the final tree against training set size
    plot_results(split_ratios, results_acc, "Mean Accuracy", "Mean Accuracy vs. Training Set Size")
    plot_results(split_ratios, results_size, "Mean Number of Nodes", "Mean Number of Nodes vs. Training Set Size")


if __name__ == "__main__":
    main()
    
    
    
