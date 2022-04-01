import scipy.io as sio
import math
import numpy as np
import matplotlib.pyplot as plt


def get_data(file):
    data = sio.loadmat(file)
    return data['data'], data['labels'], data['test_data'], data['test_labels']


def learn(data, labels, W, eta, first):
    jump = 50
    if first == 12650:
        jump = 14

    for i in range(first, first + jump):
        curr_example = data[:, i]
        p_1 = 1 / (1 + math.exp(np.dot(-1 * W, curr_example)))
        output_y = np.random.binomial(1, p_1, 1)
        if output_y == labels[0, i]:
            R = 1
        else:
            R = 0

        e_t = (output_y - p_1) * data[:, i]
        W = W + (eta * R * e_t)

    return W


def check_accuracy(W, test_data, test_labels):
    correct = 0
    num_of_tests = len(test_data[0, :])
    for i in range(num_of_tests):
        p_1 = 1 / (1 + math.exp(np.dot(-1 * W, test_data[:, i])))
        output_y = np.random.binomial(1, p_1, 1)
        if output_y == test_labels[0, i]:
            correct = correct + 1

    return correct/num_of_tests


if __name__ == '__main__':
    data, labels, test_data, test_labels = get_data("ex6_data.mat")
    W = np.random.normal(0, 0.01, size=(1, 784))
    accuracy_arr = []
    counter = 0
    counter_arr = []

    while(counter < len(data[1, :])):
        W = learn(data, labels, W, 0.01, counter)
        accuracy_arr.append(check_accuracy(W, test_data, test_labels))
        counter_arr.append(counter)
        counter = counter + 50

    plt.plot(counter_arr, accuracy_arr)
    W = W.reshape(28, 28)
    plt.imshow(W)
    plt.show()

