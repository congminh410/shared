import numpy as np
from sklearn.metrics import accuracy_score

np.set_printoptions(precision=3)


class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        best_entropy = float("inf")
        n_samples = len(y)

        for i in range(n_samples):
            if i == 0:
                X_pre = 0
            else:
                X_pre = X[i - 1]
            if X[i] != X_pre:  # Skip duplicates
                threshold = round((X[i] + X_pre) / 2, 2)  # Midpoint threshold
                left_indices = np.where(X <= threshold)[0]
                right_indices = np.where(X > threshold)[0]
                left_labels = y[left_indices]
                right_labels = y[right_indices]
                left_entropy = self._calculate_entropy(y[left_indices])
                right_entropy = self._calculate_entropy(y[right_indices])
                total_entropy = (len(left_indices) / n_samples) * left_entropy + (
                    len(right_indices) / n_samples
                ) * right_entropy

                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    self.threshold = threshold
                    self.left_class = self._get_majority_class(left_labels)
                    self.right_class = self._get_majority_class(right_labels)
                    if self.left_class is None:
                        self.left_class = self.right_class
                    if self.right_class is None:
                        self.right_class = self.left_class

    def predict(self, X):
        return np.where(X > self.threshold, self.right_class, self.left_class)

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _get_majority_class(self, ar):
        if len(ar) == 0:
            return None
        unique, counts = np.unique(ar, return_counts=True)
        return unique[np.argmax(counts)]


def bootstrap_sample(X, y, n_samples, replace=True):
    # Create training set Di by sampling from D
    indices = np.random.choice(X.shape[0], size=n_samples, replace=replace)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    sorted_indices = np.argsort(X_bootstrap)
    X_bootstrap = X_bootstrap[sorted_indices]
    y_bootstrap = y_bootstrap[sorted_indices]

    return X_bootstrap, y_bootstrap


class AdaBoost:
    def __init__(self, base_classifier, n_estimators=50):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators  # equivalent to k
        self.alphas = []
        self.classifiers = []

    def fit(self, X, y, n_samples):

        # Initialize the weights for all N examples
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = bootstrap_sample(X, y, n_samples, replace=True)
            error = 1
            while error > 0.5:
                # Create training set Di by sampling from D
                classifier = self.base_classifier()
                classifier.fit(X_bootstrap, y_bootstrap)
                predictions = classifier.predict(X)

                # Caculate the weighted error
                error = np.sum(weights * (predictions != y)) / n_samples

            # caculate alpha
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)

            # Update weights
            weights *= np.exp(-alpha * y_bootstrap * predictions)  # y_bootstrap and prediction is array of 1 or -1
            weights /= np.sum(weights)

            self.classifiers.append(classifier)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * classifier.predict(X)
        return np.sign(predictions)


if __name__ == "__main__":

    # Create a dataset
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    y = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])

    print("X: {}".format(X.reshape(1, -1)))
    print("Y: {}".format(y))

    adaboost = AdaBoost(base_classifier=DecisionStumpClassifier, n_estimators=10)
    adaboost.fit(X, y, n_samples=X.shape[0])  # Training

    print("Result")
    print("Y True: {}".format(y))
    y_pred = adaboost.predict(X)
    print("Y Prediction: {}".format(y_pred))

    # Evaluate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")