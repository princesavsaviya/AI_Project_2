import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
import sys, os

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


def forward_selection(X, y, k):
    n_features = X.shape[1]

    knn = KNeighborsClassifier(n_neighbors=k)
    baseline_accuracy = cross_val_score(knn, X, y, cv=LeaveOneOut(),n_jobs=-1,scoring='accuracy').mean()

    print(f"Baseline accuracy with {n_features} features: {baseline_accuracy*100:.2f}%\n")

    print("Starting forward selection...")
    selected_features = []
    prev_accuracy = 0.0
    best_accuracy = 0.0
    best_features = []
    history = []

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
        
        cureent_best_feature, current_best_accuracy = max(canididate_features, key=lambda x: x[1])
        selected_features.append(cureent_best_feature)
        select_str = ",".join(str(i+1) for i in selected_features)

        history.append((selected_features.copy(), current_best_accuracy*100))
        if current_best_accuracy < prev_accuracy:
            print()
            print(f"Best accuracy did not improve from {prev_accuracy*100:.2f}%")
            print("Continue to search for best feature set.")

        print()
        print(f"Selected feature(s): {{{select_str}}} with accuracy {current_best_accuracy*100:.2f}%\n")

        if current_best_accuracy > best_accuracy:
            best_accuracy = current_best_accuracy
            best_features = selected_features.copy()

        prev_accuracy = best_accuracy

        if quit_pressed():
            print("\nSearch interrupted by user. Returning best features found so far.")
            break

    best_str = ",".join(str(i+1) for i in best_features)
    print()
    print(f"Best feature set: {{{best_str}}} with accuracy {best_accuracy*100:.2f}%\n")
    return best_features, history    

def backward_elimination(X, y, k):
    n_features = X.shape[1]
    loo = LeaveOneOut()

    # 1) Baseline with all features
    knn_full = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    baseline_acc = cross_val_score(
        knn_full, X, y,
        cv=loo, scoring='accuracy', n_jobs=-1
    ).mean()
    print(f'Running nearest neighbor with all {n_features} features, '
          f'using "leaving-one-out" evaluation, I get an accuracy of '
          f'{baseline_acc*100:.1f}%\n')
    print("Beginning backward elimination search.\n")

    selected_features = list(range(n_features))
    prev_accuracy = baseline_acc
    best_accuracy = baseline_acc
    best_features = selected_features.copy()

    history = []

    while len(selected_features) > 1:
        candidate_features = []
        for i in selected_features:
            # Create a new feature set with the current candidate feature removed
            new_features = [f for f in selected_features if f != i]
            X_new = X[:, new_features]

            # Evaluate the model with the new feature set
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            accuracy = cross_val_score(
                knn, X_new, y,
                cv=loo, n_jobs=-1, scoring='accuracy'
            ).mean()

            feat_str = ",".join(str(f+1) for f in new_features)
            print(f'Using feature(s) {{{feat_str}}} accuracy is {accuracy*100:.1f}%')
            candidate_features.append((i, accuracy, new_features))

        # Pick the removal that yields the highest accuracy
        removed_features, accuracy, new_features = max(
            candidate_features, key=lambda x: x[1]
        )

        if accuracy < prev_accuracy:
            print("\n(Warning, Accuracy has decreased! "
                  "Continuing search in case of local maxima)\n")

        # Update selected_features to the new set before printing
        selected_features = new_features.copy()
        select_str = ",".join(str(f+1) for f in selected_features)
        print(f"Selected feature(s): {{{select_str}}} "
              f"with accuracy {accuracy*100:.2f}%\n")

        history.append((selected_features.copy(), accuracy * 100))
        prev_accuracy = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = selected_features.copy()

        if quit_pressed():
            print("\nSearch interrupted by user. Returning best features found so far.")
            break

    best_str = ",".join(str(f+1) for f in best_features)
    print(f"\nBest feature set: {{{best_str}}} "
          f"with accuracy {best_accuracy*100:.2f}%\n")
    return best_features,history

def graph_history(history, title="Feature Selection History"):

    if not history:
        print("No history to graph.")
        return

    max_len = max(len(feats) for feats, _ in history)

    labels = []
    accuracies = []
    for feats, acc in history:
        # Build label
        if not feats:
            lbl = "{}"
        elif len(feats) == max_len:
            lbl = "{All Features}"
        elif len(feats) <=3:
            lbl = "{" + ",".join(str(f) for f in feats) + "}"
        else:
            lbl = "{" + ",".join(str(f) for f in feats[:3]) + ",...}"
        labels.append(lbl)
        accuracies.append(acc)

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, marker='o')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print("Welcome to KNN Feature Selection!")
    file_name = input("Please enter the data file name (e.g., CS205_small_Data__22.txt): ")
    data = np.loadtxt(file_name, dtype=np.float64, ndmin=2)
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k = 2
    algo = int(input("Select algorithm \n 1) forward selection \n 2) backward elimination \n Enter 1 or 2 :: "))

    print("This Dataset has {} features and {} samples.".format(X.shape[1], X.shape[0]))
    print()
    print(" You can press 'q' at any time to quit the search early. Search ends after the completion of current iteration.\n")
    start_time = time.time()
    if algo == 1:
        
        selected_features,history = forward_selection(X, y, k)
        print()
        print(f"Final selected features (0-based indexing): {selected_features}")
        end_time = time.time()
        graph_history(history, title="Forward Selection History")  # Graph the history for forward selection

    elif algo == 2:
        selected_features,history = backward_elimination(X, y, k)
        print()
        print(f"Final selected features (0-based indexing): {selected_features}")
        end_time = time.time()
        graph_history(history, title="Backward Elimination History")

    print(f"Total time taken for search: {end_time - start_time:.2f} seconds")
    print("Thank you for using KNN Feature Selection!")