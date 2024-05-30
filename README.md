- Decision Trees using Scikit-learn:

Use the Banknote Authentication data attached with the assignment to
implement the following requirements:
1. Experiment with a fixed train_test split ratio: Use 25% of the samples for
training and the rest for testing.
a. Run this experiment five times and notice the impact of different
random splits of the data into training and test sets.
b. Print the sizes and accuracies of these trees in each experiment.
2. Experiment with different range of train_test split ratio: Try (30%-70%),
(40%-60%), (50%-50%), (60%-40%) and (70%-30%):
a. Run the experiment with five different random seeds for each of
split ratio.
b. Calculate mean, maximum and minimum accuracy for each split
ratio and print them.
c. Print the mean, max and min tree size for each split ratio.
d. Draw two plots: 1) shows mean accuracy against training set size
and 2) the mean number of nodes in the final tree against training
set size.

Note: the size of the tree is number of its nodes.

- KNN:

Use the Banknote Authentication data to implement your own simple KNN
classifier using python, (Don’t use any built-in functions):
1. Divide your data into 70% for training and 30% for testing.
2. Each feature column should be normalized separately from all other
features. Specifically, for both training and test objects, each feature
should be transformed using the function: f(v) = (v - mean) / std, using the
mean and standard deviation of the values of that feature column on the
training data.
3. If there is a tie in the class predicted by the k-nn, then among the classes
that have the same number of votes, you should pick the one that comes
first in the Train file.
4. Use Euclidean distance to compute distances between instances.
5. Experiment with different values of k=1,2,3, ...., 9 and print the following:
• The value of k used for the test set.
• The number of correctly classified test instances.
• The total number of instances in the test set. • The accuracy.
