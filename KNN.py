import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt("CS205_small_Data__22.txt", dtype=np.float64, ndmin=2)
y = data[:, 0].astype(np.int64)
X = data[:, 1:]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2, algorithm='auto', metric='euclidean')
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy of KNN classifier with k=2: {accuracy*100:.2f}")