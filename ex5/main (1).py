import numpy as np
import matplotlib.pyplot as plt
import random
import math


def q1(p, f, sigmaX, num_of_examples):
    examples = np.zeros(shape=(2, num_of_examples))
    examples[0, :] = np.random.uniform(-1, 1, size=(1, num_of_examples))
    for i in range(num_of_examples):
        eps = np.random.normal(0, sigmaX, size=(1, 1))
        t = random.random()
        if t < p:
            examples[1, i] = (math.sin(f*examples[0, i])) + eps
        else:
            examples[1, i] = np.random.uniform(-1, 1)

    return examples


def q2(dimension, K, p, f, sigmaX, num_of_examples):
    examples = q1(p, f, sigmaX, num_of_examples)
    start_prototypes = np.random.uniform(-1, 1, (dimension, K))
    return examples, start_prototypes


def q3(examples, prototypes, num_of_examples, sigma, K):
    ks = np.arange(0, K)
    for i in range(num_of_examples):
        xn = examples[:, i]
        k = np.argmin(np.linalg.norm(xn[:, np.newaxis] - prototypes, axis=0))
        pifunc = np.exp(-1*(1/(2*sigma*sigma))*(ks-k)*(ks-k))
        pifunc = pifunc/sum(pifunc)
        prototypes = prototypes + (xn[:, np.newaxis] - prototypes)*pifunc

    return prototypes


if __name__ == '__main__':
    examples, prototypes = q2(2, 100, 0.95, 4, 0.1, 20000)
    prototypes = q3(examples, prototypes, 20000, 4, 100)
    plt.scatter(examples[0, :], examples[1, :], alpha=0.05)
    plt.scatter(prototypes[0, :], prototypes[1, :], c='r', s=30)
    plt.plot(prototypes[0, :], prototypes[1, :], '--', c='r')
    plt.show()