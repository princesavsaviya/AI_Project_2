import numpy as np
import pandas as pd

class KNN:
    def __init__(self,k=1):
        """
        k : number of nearest neighbors to consider
        """
        self.k = k
    
    def fit(self, X, y):
        """
        X : training data, shape (n_samples, n_features)
        y : target values, shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        X : test data, shape (n_samples, n_features)
        Returns predicted labels for each sample in X
        """
        predictions = []
        for x in X:
            # Compute distances from x to all training samples
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Extract the labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            # Predict the label by majority vote
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)
    

def load_data(file_path,max_features=64,max_instances = 2048):
    data = np.loadtxt(file_path,dtype=np.float64,ndmin=2,max_rows=max_instances)

    y = data[:, 0].astype(np.int64)
    X = data[:, 1:1+max_features]

    return X, y

def standardize(X):
    """
    Standardize the features by removing the mean and scaling to unit variance.
    X : input data, shape (n_samples, n_features)
    Returns standardized data
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_standardized = (X - mean) / std
    return X_standardized

if __name__ == "__main__":
    X,y = load_data("CS205_small_Data__22.txt")
    X = standardize(X)
    n = X.shape[0]
    idx = np.random.permutation(n)
    split_index = int(0.8 * n)  # 80% for training, 20% for testing
    X_train, X_test = X[idx[:split_index]], X[idx[split_index:]]
    y_train, y_test = y[idx[:split_index]], y[idx[split_index:]]

    # Create and fit the KNN model
    knn = KNN(k=3)  # You can change k to any value you want
    knn.fit(X_train, y_train)
    # Predict on the training data (for demonstration purposes)
    predictions = knn.predict(X_test)
    print("Predictions on training data:")
    print(predictions[:10])  # Show first 10 predictions
    # Check accuracy on training data
    accuracy = np.mean(predictions == y_test)
    print(f"Training accuracy: {accuracy * 100:.2f}%")