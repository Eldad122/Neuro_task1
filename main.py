import numpy as np
import random
from sklearn.model_selection import train_test_split

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

