import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    X = np.array([line[:-1] for line in data], dtype=float)
    y = np.array([line[-1] for line in data])
    return X, y



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def get_neighbors(X_train, y_train, x_test, k):
    distances = []
    for i, x_train in enumerate(X_train):
        dist = euclidean_distance(x_train, x_test)
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append((y_train[distances[i][0]], distances[i][1]))
    return neighbors

def predict_classification(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[0]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def classify_test_set(X_train, y_train, X_test, k):
    y_pred = []
    for x_test in X_test:
        neighbors = get_neighbors(X_train, y_train, x_test, k)
        prediction = predict_classification(neighbors)
        y_pred.append(prediction)
    return y_pred

def main():
    train_file_path = input("Enter the path to the training file: ")
    X_train, y_train = load_data(train_file_path)

    k = int(input("Enter the number of nearest neighbors (k): "))

    while True:
        print("\nChoose an option:")
        print("a) Classification of all observations from the test set")
        print("b) Classification of an observation given by the user")
        print("c) Change k")
        print("d) Exit")

        option = input("Option: ")

        if option == 'a':
            test_file_path = input("Enter the path to the test file: ")
            X_test, y_test = load_data(test_file_path)
            y_pred = classify_test_set(X_train, y_train, X_test, k)
            acc = accuracy(y_test, y_pred)
            print("Predicted labels for test set:")
            for i in range(len(y_test)):
                print("Observation", i+1, "- Predicted Label:", y_pred[i])
            print("Accuracy:", acc)

        elif option == 'b':
            observation = np.array([float(x) for x in input("Enter observation attributes separated by comma: ").split(',')])
            prediction = predict_classification(get_neighbors(X_train, y_train, observation, k))
            print("Predicted label:", prediction)

        elif option == 'c':
            k = int(input("Enter the number of nearest neighbors (k): "))

        elif option == 'd':
            print("Exiting...")
            break

        else:
            print("Invalid option. Please choose again.")

if __name__ == "__main__":
    main()
