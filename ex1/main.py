import numpy as np
import matplotlib.pyplot as plt
import math

def q1(N, P, f):
    mem_matrix = np.zeros(shape=(N, P))  # memory matrix, N neurons, P memory patterns
    J = np.zeros(shape=(N, N))  # synapses matrix, N neurons

    fill_mem_vec = np.vectorize(fill_mem)
    mem_matrix = fill_mem_vec(mem_matrix, f)

    mem_matrix_minus_f = mem_matrix - f

    J = (1 / (f * (1 - f) * N)) * np.matmul(mem_matrix_minus_f, np.transpose(mem_matrix_minus_f))
    np.fill_diagonal(J, 0)  # set diagonal to be zeros

    return mem_matrix, J


def fill_mem(a, f):
    return np.random.binomial(1, f)  # bernoulli w.p f


def q2(J, s, T):
    prev_s = np.copy(s)
    while 1:
        for idx in range(s.size):
            h_i = np.dot(s, J[idx, :])
            s[idx] = np.heaviside(h_i - T, 0)  # theta function on hi-T

        if np.array_equal(prev_s, s):
            return s

        prev_s = np.copy(s)  # we need to keep updating


def q3(J, s, T):
    old_s = np.copy(s)
    new_s = q2(J, old_s, T)
    num_changed_neurons = np.sum(new_s != s)
    num_neurons = np.size(s)
    return num_changed_neurons/num_neurons


def q4(f):
    alpha = 0.02
    T = 0.5 - f
    N = 1000
    mean_arr = []
    alpha_arr = []
    while alpha <= 0.8:
        m = []
        P = math.ceil(N * alpha)
        for i in range(5):
            mem_matrix, J = q1(N, P, f)
            m.append(q3(J, mem_matrix[:, 0], T))
        mean_arr.append(np.mean(m))
        alpha_arr.append(alpha)
        alpha += 0.04
    return mean_arr, alpha_arr


def q5():
    f = [0.1, 0.2, 0.3]
    plt.xlabel('alpha')
    plt.ylabel('error rate')
    plt.title('Error rate per alpha')
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mean_arr, alpha_arr = q4(f[i])
        plt.plot(alpha_arr, mean_arr, label="f =" + str(f[i]), color=colors[i])
    plt.legend()
    plt.show()
