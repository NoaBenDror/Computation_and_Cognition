import numpy as np
import matplotlib.pyplot as plt


def q1(s, a):
    next_s = 1
    r = 1

    if s == 1:
        if a == 1:
            next_s = 1
            r = 0
        elif a == 2:
            next_s = np.random.choice(a=[1, 2], p=[0.2, 0.8])
            r = 1
    elif s == 2:
        if a == 1:
            next_s = 2
            r = 2
        elif a == 2:
            next_s = 1
            r = 0

    return next_s, r


def q2():
    curr_s = 1
    eta = 0.01
    gama = 0.5
    V_home_s = np.zeros(3000)
    V_out_s = np.zeros(3000)

    for i in range(1, 3000):
        curr_a = np.random.choice(a=[1, 2], p=[0.5, 0.5])
        new_s, r = q1(curr_s, curr_a)

        V_home = V_home_s[i - 1]
        V_out = V_out_s[i - 1]

        if (curr_s == 1) and (new_s == 1):
            V_home_s[i] = V_home + eta * (r + gama * V_home - V_home)
            V_out_s[i] = V_out
        elif (curr_s == 1) and (new_s == 2):
            V_home_s[i] = V_home + eta * (r + gama * V_out - V_home)
            V_out_s[i] = V_out
        elif (curr_s == 2) and (new_s == 1):
            V_home_s[i] = V_home
            V_out_s[i] = V_out + eta * (r + gama * V_home - V_out)
        elif (curr_s == 2) and (new_s == 2):
            V_home_s[i] = V_home
            V_out_s[i] = V_out + eta * (r + gama * V_out - V_out)

        curr_s = new_s

    v_p_home = np.ones(3000) * (23/19)
    v_p_out = np.ones(3000) * (33/19)
    plt.plot(V_home_s)
    plt.plot(V_out_s)
    plt.plot(v_p_home)
    plt.plot(v_p_out)
    plt.show()


def q3():
    eta = 0.1
    gama = 0.5
    curr_q = np.zeros(shape=(2, 2))
    curr_s = np.random.choice(a=[1, 2], p=[0.5, 0.5])
    Vs = np.zeros(shape=(2, 3000))

    for i in range(3000):
        curr_a = np.random.choice(a=[1, 2], p=[0.5, 0.5])
        new_s, r = q1(curr_s, curr_a)
        curr_q[curr_s-1][curr_a-1] = curr_q[curr_s-1][curr_a-1] + eta * (r + gama * np.max(curr_q[new_s-1]) - curr_q[curr_s-1][curr_a-1])
        curr_s = new_s
        Vs[0, i] = np.max(curr_q[0])
        Vs[1, i] = np.max(curr_q[1])

    real_v_star1 = np.ones(3000) * (26 / 9)
    real_v_star2 = np.ones(3000) * 4
    plt.plot(Vs[0])
    plt.plot(Vs[1])
    plt.plot(real_v_star1)
    plt.plot(real_v_star2)
    plt.show()


def helper_value_iteration(V_home, V_out):
    s1a1 = 0.5 * V_home
    s1a2 = 1 + 0.5 * (0.2 * V_home + 0.8 * V_out)
    s2a1 = 2 + 0.5 * V_out
    s2a2 = 0.5 * V_home
    new_V_home = max(s1a1, s1a2)
    new_V_out = max(s2a1, s2a2)
    return new_V_home, new_V_out


def value_iteration():
    V_home = np.zeros(3000)
    V_out = np.zeros(3000)

    for i in range(1, 5000):
        V_home[i], V_out[i] = helper_value_iteration(V_home[i-1], V_out[i-1])
        if V_home[i] - V_home[i-1] < 0.001 and V_out[i] - V_out[i-1] < 0.001:
            break
    return V_home[i], V_out[i]


if __name__ == '__main__':
    print(value_iteration())
    q2()
    q3()
