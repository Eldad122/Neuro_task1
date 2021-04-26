import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from AdalineAlgo import CustomAdaline


def functionA(Xi):
    if Xi[0] > 0.5 and Xi[1] > 0.5:
        return 1
    else:
        return -1

def functionB(Xi):
    if 0.5 <= (Xi[0] ** 2 + Xi[1] ** 2) <= 0.75:
        return 1
    else:
        return -1

def data(data_part: int):
    x_data = []
    y_data = []
    for i in range(1000):
        x1 = float(random.randint(-100, 100))
        x2 = float(random.randint(-100, 100))
        x1 /= 100
        x2 /= 100
        temp_x = np.array([x1, x2])
        if data_part == 1:
            temp_y = functionA(temp_x)
        else:
            temp_y = functionB(temp_x)
        x_data.append(temp_x)
        y_data.append(temp_y)
    return np.array(x_data), np.array(y_data)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #
    # Load the data set
    #
    X, y = data(1)
    #
    # Create training and test split
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #
    # Instantiate CustomPerceptron
    #
    adaline = CustomAdaline(n_iterations=10)
    #
    # Fit the model
    #
    adaline.fit(X_train, y_train)
    #
    # Score the model
    #
    print(adaline.score(X_test, y_test))
    plot_decision_regions(X, y, classifier=adaline)

    plt.title('Adaptive Linear Neuron - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()