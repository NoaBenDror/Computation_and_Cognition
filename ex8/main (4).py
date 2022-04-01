import numpy as np
import time
import os
import matplotlib.pyplot as plt


def data_collection():
    input_choices = np.zeros(100)
    rewards = np.zeros(100)

    name = input("Please write your name: ")
    for i in range(100):
        choice = input("Choose 1 or 0: \n")
        while choice != '0' and choice != '1':  # check valid input
            choice = input("Choose 1 or 0: \n")

        input_choices[i] = choice
        if choice == '1':
            r = np.random.choice(a=[0, 1], p=[0.4, 0.6])
        else:
            r = np.random.choice(a=[0, 1], p=[0.65, 0.35])
        rewards[i] = r
        print("You get " + str(r) + " points!")
        time.sleep(1)
        os.system('cls')

    np.savetxt(name + " choices.csv", input_choices, fmt='%1.0f', delimiter=',')
    np.savetxt(name + " rewards.csv", rewards, fmt='%1.0f', delimiter=',')

    return input_choices, rewards


def data_presentation(input_choices):
    trials = [0, 0]
    for i in range(20):
        if input_choices[i] == 1:
            trials[0] += 1
    for i in range(80, 100):
        if input_choices[i] == 1:
            trials[1] += 1

    trials[0] = trials[0] / 20
    trials[1] = trials[1] / 20
    plt.bar(["1", "2"], trials)
    plt.ylim(0, 1)
    plt.show()


def reinforce_calc_log_likelihood(input_choices, rewards, eta):
    likelihood = 0
    p = 0.5
    w = 0
    for t in range(100):
        y = input_choices[t]
        r = rewards[t]
        if y == 1:
            trial_likelihood = p
        else:
            trial_likelihood = (1 - p)
        likelihood += np.log(trial_likelihood)

        w = w + (eta * r * (y - p))
        p = 1 / (1 + np.exp(-1 * w))

    return likelihood


def reinforce_find_optimal_eta(input_choices, rewards):
    max_likelihood = reinforce_calc_log_likelihood(input_choices, rewards, 0)
    optimal_eta = 0
    for eta in np.arange(0.005, 1.0, 0.005):
        current_likelihood = reinforce_calc_log_likelihood(input_choices, rewards, eta)
        if current_likelihood > max_likelihood:
            max_likelihood = current_likelihood
            optimal_eta = eta

    return optimal_eta


def reinforce_simulation(eta):
    choices = []
    rewards = []
    p = 0.5
    w = 0
    for t in range(100):
        y = np.random.choice([0, 1], p=[1 - p, p])
        choices.append(y)
        if y == 1:
            r = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            r = np.random.choice([0, 1], p=[0.65, 0.35])
        rewards.append(r)

        w = w + (eta * r * (y - p))
        p = 1 / (1 + np.exp(-1 * w))

    return choices, rewards


def run_reinforce_simulations(eta):
    eta_arr = []
    for t in range(100):
        choices, rewards = reinforce_simulation(eta)
        current_eta = reinforce_find_optimal_eta(choices, rewards)
        eta_arr.append(current_eta)
    return eta_arr


def TD_calc_log_likelihood(input_choices, rewards, eta):
    likelihood = 0
    p = 0.5
    V = [0, 0]
    for t in range(100):
        y = int(input_choices[t])
        r = int(rewards[t])
        if y == 1:
            trial_likelihood = p
        else:
            trial_likelihood = (1 - p)

        likelihood += np.log(trial_likelihood)
        V[y] = V[y] + (eta * (r - V[y]))
        p = np.exp(V[1]) / (np.exp(V[1]) + np.exp(V[0]))

    return likelihood


def TD_find_optimal_eta(input_choices, rewards):
    max_likelihood = TD_calc_log_likelihood(input_choices, rewards, 0)
    optimal_eta = 0
    for eta in np.arange(0.005, 1.0, 0.005):
        current_likelihood = TD_calc_log_likelihood(input_choices, rewards, eta)
        if current_likelihood > max_likelihood:
            max_likelihood = current_likelihood
            optimal_eta = eta

    return optimal_eta


def TD_simulation(eta):
    choices = []
    rewards = []
    V = [0, 0]
    p = 0.5
    for t in range(100):
        y = np.random.choice([0, 1], p=[1 - p, p])
        choices.append(y)
        if y == 1:
            r = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            r = np.random.choice([0, 1], p=[0.65, 0.35])
        rewards.append(r)
        V[y] = V[y] + (eta * (r - V[y]))
        p = np.exp(V[1]) / (np.exp(V[1]) + np.exp(V[0]))

    return choices, rewards


def run_TD_simulation(eta):
    eta_arr = []
    for t in range(100):
        choices, rewards = TD_simulation(eta)
        current_eta = TD_find_optimal_eta(choices, rewards)
        eta_arr.append(current_eta)
    return eta_arr


mai_choices = np.loadtxt("mai choices.csv", delimiter=',')
mai_rewards = np.loadtxt("mai rewards.csv", delimiter=',')
mai_reinforce_eta = reinforce_find_optimal_eta(mai_choices, mai_rewards)
mai_TD_eta = TD_find_optimal_eta(mai_choices, mai_rewards)

print('mai reinforce eta: ', mai_reinforce_eta)
print('mai TD eta: ', mai_TD_eta)

guy_choices = np.loadtxt("guy choices.csv", delimiter=',')
guy_rewards = np.loadtxt("guy rewards.csv", delimiter=',')
guy_reinforce_eta = reinforce_find_optimal_eta(guy_choices, guy_rewards)
guy_TD_eta = TD_find_optimal_eta(guy_choices, guy_rewards)

print('guy reinforce eta: ', guy_reinforce_eta)
print('guy TD eta: ', guy_TD_eta)


print('mai reinforce start')
mai_reinforce_etas_lst = run_reinforce_simulations(mai_reinforce_eta)
print('mai TD start')
mai_TD_etas_lst = run_TD_simulation(mai_TD_eta)

print('guy reinforce start')
guy_reinforce_etas_lst = run_reinforce_simulations(guy_reinforce_eta)
print('guy TD start')
guy_TD_etas_lst = run_TD_simulation(guy_TD_eta)

plt.hist(mai_reinforce_etas_lst, label='reinforce', bins=20)
plt.hist(mai_TD_etas_lst, label='TD', bins=20)
plt.title('Estimated eta values for mai')
plt.legend()
plt.show()

plt.hist(guy_reinforce_etas_lst, label='reinforce', bins=20)
plt.hist(guy_TD_etas_lst, label='TD', bins=20)
plt.title('Estimated eta values for guy')
plt.legend()
plt.show()