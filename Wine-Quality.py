import numpy as np
from KNN import forward_selection,graph_history
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sys,os

# Check for quit key press
if os.name == 'nt':
    import msvcrt
    def quit_pressed():
        # Windows: check if a key was hit and if it was 'q'
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch.lower() == 'q'
        return False
else:
    import select
    def quit_pressed():
        # Unix: non-blocking select on stdin
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            line = sys.stdin.readline()
            return line.strip().lower() == 'q'
        return False
    
red_wine = pd.read_csv('wine+quality\winequality-red.csv', sep=';')
white_wine = pd.read_csv('wine+quality\winequality-white.csv', sep=';')

# Combine datasets and create labels
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
wine_data['quality'] = wine_data['quality'].astype(int)

# Features and labels
X = wine_data.drop(columns=['quality']).values
y = wine_data['quality'].values

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

default_rate = np.bincount(y_train).argmax()  # Most common class in training set
print(f"Default classification rate (most common class): {default_rate} ({np.mean(y_train == default_rate) * 100:.2f}%)\n")

# Perform forward selection with KNN
k = 10  # Number of neighbors for KNN as there are 10 classes in the dataset
selected_features, history = forward_selection(X_train, y_train, k)

# Evaluate the best feature set on the test set
if selected_features:
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_selected, y_train)

    test_accuracy = knn.score(X_test_selected, y_test)
    print(f"Test accuracy with selected features: {test_accuracy * 100:.2f}%\n")

    # Graph the history of feature selection
    graph_history(history)