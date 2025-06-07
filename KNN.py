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

    y = data[:, 0]
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
    print("Data loaded and standardized.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"First 5 labels: {y[:5]}")
    print(f"First 5 samples: {X[:5]}")

    # Create and fit the KNN model
    knn = KNN(k=3)  # You can change k to any value you want
    knn.fit(X, y)
    # Predict on the training data (for demonstration purposes)
    predictions = knn.predict(X)
    print("Predictions on training data:")
    print(predictions[:10])  # Show first 10 predictions
    # Check accuracy on training data
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    