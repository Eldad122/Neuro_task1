import numpy as np
import matplotlib.pyplot as plt
import kohonen


def isInHand(x, y):  # 4 fingers
    if y < 0.4:
        return True
    elif 0 < x < 0.25 and y < 0.7:
        return True
    elif 0.25 < x < 0.5 and y < 0.8:
        return True
    elif 0.5 < x < 0.75 and y < 0.9:
        return True
    elif 0.75 < x < 1 and y < 0.8:
        return True
    return False


def isIn3Hand(x, y):  # 3 fingers
    if y < 0.4:
        return True
    # elif 0 < x < 0.25 and y < 0.7:
    #    return True
    elif 0.25 < x < 0.5 and y < 0.8:
        return True
    elif 0.5 < x < 0.75 and y < 0.9:
        return True
    elif 0.75 < x < 1 and y < 0.8:
        return True
    return False


def sample1():
    size = 1000
    x = np.random.uniform(0, 1, size)
    y = np.random.uniform(0, 1, size)
    for i in range(size):
        if not isInHand(x[i], y[i]):
            while not isInHand(x[i], y[i]):
                x[i] = np.random.uniform(0, 1)
                y[i] = np.random.uniform(0, 1)
    step_max = 3000
    radius = 10
    alpha = 0.3
    kohonen.draw(x, y, 1, radius, alpha, is_line=False)
    kohonen.train(x, y, step_max, radius, alpha)


def sample2():
    size = 1000
    x = np.random.uniform(0, 1, size)
    y = np.random.uniform(0, 1, size)
    for i in range(size):
        if not isIn3Hand(x[i], y[i]):
            while not isIn3Hand(x[i], y[i]):
                x[i] = np.random.uniform(0, 1)
                y[i] = np.random.uniform(0, 1)
    step_max = 3000
    radius = 10
    alpha = 0.3
    kohonen.draw(x, y, 1, radius, alpha, is_line=False)
    kohonen.train(x, y, step_max, radius, alpha)


if __name__ == '__main__':
    chosen = input("Choose a sample: ")
    if chosen == '1':
        sample1()
    elif chosen == '2':
        sample2()
