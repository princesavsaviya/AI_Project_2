import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

def forward_selection(X, y, k):
    n_features = X.shape[1]
    knn = KNeighborsClassifier(n_neighbors=k)
    baseline_accuracy = cross_val_score(knn, X, y, cv=LeaveOneOut(),n_jobs=-1,scoring='accuracy').mean()

    print(f"Baseline accuracy with {n_features} features: {baseline_accuracy*100:.2f}%\n")

    print("Starting forward selection...")
    selected_features = []
    prev_accuracy = 0.0

    while True:
        canididate_features = []
        for i in range(n_features):
            if i in selected_features:
                continue
            # Create a new feature set with the current candidate feature added
            new_features = selected_features + [i]
            X_new = X[:, new_features]

            # Evaluate the model with the new feature set
            knn = KNeighborsClassifier(n_neighbors=k)
            accuracy = cross_val_score(knn, X_new, y, cv=LeaveOneOut(), n_jobs=-1, scoring='accuracy').mean()

            feat_str = ",".join(str(i+1) for i in new_features)  # +1 for 1-based indexing
            print(f'Using feature(s) {{{feat_str}}} accuracy is {accuracy*100:.1f}%')
            canididate_features.append((i, accuracy))

        if not canididate_features:
            print("No more candidate features to add.")
            break
        
        best_feature, best_accuracy = max(canididate_features, key=lambda x: x[1])
        selected_features.append(best_feature)
        select_str = ",".join(str(i+1) for i in selected_features)

        if best_accuracy < prev_accuracy:
            print(f"Stopping forward selection. Best accuracy did not improve from {prev_accuracy*100:.2f}%")
            print("Continue to search for best feature set.")

        print(f"Selected feature(s): {{{select_str}}} with accuracy {best_accuracy*100:.2f}%\n")
        prev_accuracy = best_accuracy

    return selected_features    

data = np.loadtxt("CS205_small_Data__22.txt", dtype=np.float64, ndmin=2)
y = data[:, 0].astype(np.int64)
X = data[:, 1:]

scaler = StandardScaler()
X = scaler.fit_transform(X)

selected_features = forward_selection(X, y, k=3)
print(f"Final selected features (0-based indexing): {selected_features}")