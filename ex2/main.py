import numpy as np
import matplotlib.pyplot as plt


def vertic_vec(v):
    if v[0] == 0:
        return [1, 0]
    elif v[1] == 0:
        return [0, 1]
    else:
        return [1, (-1*v[0]) / v[1]]


def q1(P, y, N):

    w = np.ones(N)
    keep_updating = 1

    while keep_updating:
        keep_updating = 0
        for i in range(len(P[0])):
            if (y[i] * np.dot(w, P[:, i])) <= 0:
                w = w + (y[i] * P[:, i])
                keep_updating = 1
    return w


def q2(P):
    P_mat = np.random.uniform(-10, 10, (2, P))
    y = np.zeros(P)
    ax = plt.gca()
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    colors = []

    for i in range(P):
        if P_mat[0, i] >= P_mat[1, i]:
            colors.append('blue')
            y[i] = 1
        else:
            colors.append('red')
            y[i] = -1

    return P_mat, y, colors


def q3():
    P_mat, y, colors = q2(1000)
    plt.scatter(P_mat[0, :], P_mat[1, :], s=0.5, c=colors)
    w = q1(P_mat, y, 2)
    norm = np.linalg.norm(w)
    normal_w = w / norm
    vertical_vec = vertic_vec(normal_w)
    plt.arrow(0, 0, normal_w[0], normal_w[1], head_length=5)
    plt.arrow(0, 0, vertical_vec[0], vertical_vec[1], color='pink', head_length=12)
    plt.arrow(0, 0, -1*vertical_vec[0], -1*vertical_vec[1], color='pink', head_length=12)
    plt.show()

    return P_mat, y, colors


def q4():
    P_sizes = [500, 200, 150, 100, 50, 30, 20, 5]
    M = 100
    optimal_w = [1, -1]
    optimal_w = (optimal_w / np.linalg.norm(optimal_w))
    err_av_arr = np.zeros(len(P_sizes))

    for i in range(len(P_sizes)):
        for j in range(M):
            P_mat, y, colors = q2(P_sizes[i])
            w = q1(P_mat, y, 2)
            w = w / np.linalg.norm(w)
            error = np.rad2deg(np.arccos(np.dot(w, optimal_w)))
            err_av_arr[i] = err_av_arr[i] + error
        err_av_arr[i] = err_av_arr[i] / M

    plt.xlabel('P')
    plt.ylabel('Error')
    plt.title('The average error as a function of P')
    plt.plot(P_sizes, err_av_arr)
    plt.show()