# -*- coding: utf-8 -*-

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt
import random
### Package Imports ###


n_states = 3 # 0 is Safely Employed (SE), 1 is PIP, 2 is Unemployed (UE)
n_actions = 2 # 0 is Code, 1 is Netflix

t_p = np.zeros((n_states, n_actions, n_states))

# Transition Probabilities: These are represented as a 3-dimensional array
# t_p[s_1, a, s_2] = p indicates that beginning from state s_1 and taking action a will result in state s_2 with probability p
t_p[0, 0, 0] = 1        # t_p[SE, Code, SE]      = 1
t_p[0, 1, 0] = 1 / 4    # t_p[SE, Netflix, SE]   = 1 / 4
t_p[0, 1, 1] = 3 / 4    # t_p[SE, Netflix, PIP]  = 3 / 4
t_p[1, 0, 0] = 1 / 4    # t_p[PIP, Code, SE]     = 1 / 4
t_p[1, 0, 1] = 3 / 4    # t_p[PIP, Code, PIP]    = 3 / 4
t_p[1, 1, 1] = 7 / 8    # t_p[PIP, Netflix, PIP] = 7 / 8
t_p[1, 1, 2] = 1 / 8    # t_p[PIP, Netflix, UE]  = 1 / 8
# all other transition probabilities are 0

R_NETFLIX = 10
R_CODING = 4

SE = 0
PIP = 1
UE = 2

CODING = 0
NETFLIX = 1

def policy_evaluation(gamma, current_policy):
    
    if current_policy[SE] == NETFLIX:
        a = 1 - (gamma * t_p[0, 1, 0])
        b = -1 * gamma * t_p[0, 1, 1]
        c = R_NETFLIX
    else: # current_policy[SE] == CODING
        a = 1
        b = 0
        c = R_CODING / (1 - (gamma * t_p[0, 0, 0]))
        
    if current_policy[PIP] == NETFLIX:
        d = 0
        e = 1 - (gamma * t_p[1, 1, 1])
        f = R_NETFLIX
    else: # current_policy[PIP] == CODING
        d = -1 * gamma * t_p[1, 0, 0]
        e = 1 - (gamma * t_p[1, 0, 1])
        f = R_CODING 

    x = np.array([[a, b], [d, e]])   
    y = np.array([c, f])

    return np.linalg.solve(x, y)

def policy_iteration(gamma):
    """
    We find the optimal policy under the constraints of discount factor gamma, which is given as a parameter.
    Relevant variables and the transition probabilities are defined as globals above.
    """

    # theta = 1e-5 # define a theta that determines if the change in utilities from iteration to iteration is "small enough"
    
    current_policy = np.ones(n_states, dtype=int) # define your policy, which begins as Netflix regardless of state
    
    while True:
        # Policy Evaluation
        policy_values = policy_evaluation(gamma, current_policy)

        U1SE = policy_values[0]
        U1PIP = policy_values[1]

        # Policy Iteration
        new_policy = np.ones(n_states, dtype=int)

        Pi1SE_netflix = R_NETFLIX + gamma * ((t_p[0, 1, 0] * U1SE) + (t_p[0, 1, 1] * U1PIP))
        Pi1SE_coding = R_CODING + gamma * ((t_p[0, 0, 0] * U1SE) + (0 * U1PIP))

        if Pi1SE_coding > Pi1SE_netflix:
            new_policy[SE] = CODING
        else:
            new_policy[SE] = NETFLIX

        Pi1PIP_netflix = R_NETFLIX + gamma * ((t_p[1, 1, 2] * 0) + (t_p[1, 1, 1] * U1PIP))
        Pi1PIP_coding = R_CODING + gamma * ((t_p[1, 0, 0] * U1SE) + (t_p[1, 0, 1] * U1PIP))

        if Pi1PIP_coding > Pi1PIP_netflix:
            new_policy[PIP] = CODING
        else:
            new_policy[PIP] = NETFLIX
            
        # Policy Change check
        if np.array_equal(new_policy, current_policy):
            break

        current_policy = new_policy

    return current_policy

    
def simulation(iterations, gamma, optimal_policy):

    current_state = SE
    total_utility = 0
    total_utility_after_i_iterations = []

    for i in iterations:

        random_num = random.random()

        if current_state == SE:
            if optimal_policy[SE] == NETFLIX:
                if random_num > 1/4:
                    current_state = PIP
                total_utility += ((gamma**i) * R_NETFLIX)
            else: # optimal_policy[SE] == CODING
                total_utility += ((gamma**i) * R_CODING)
        
        elif current_state == PIP:
            if optimal_policy[PIP] == NETFLIX:
                if random_num > 7/8:
                    current_state = UE
                total_utility += ((gamma**i) * R_NETFLIX)
            else: # optimal_policy[PIP] == CODING
                if random_num > 3/4:
                    current_state = SE
                total_utility += ((gamma**i) * R_CODING)

        total_utility_after_i_iterations.append(total_utility)
    
    return total_utility_after_i_iterations


def value_plots(gamma09, optimal_policy_09, gamma08, optimal_policy_08):
    """
    The plots indicate the cumulative utility summed across all states across iterations. More specifically, the y-val
    indicates the total amount of utility acumulated across the states and actions as the iterations progress. This means
    we keep track of what policies we have at every iteration.
    """
    
    iterations = range(0, 50)

    p1_vals = simulation(iterations, gamma09, optimal_policy_09)
    p2_vals = simulation(iterations, gamma08, optimal_policy_08)

    plt.plot(iterations, p1_vals, label="Policies for gamma = 0.9")
    plt.plot(iterations, p2_vals, label="Policies for gamma = 0.8")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Utility Value")
    plt.legend()
    plt.title("Cumulative Utility Values over Time")
    plt.show()

if __name__ == "__main__":

    optimal_policy_09 = policy_iteration(0.9)
    optimal_policy_08 = policy_iteration(0.8)
    value_plots(0.9, optimal_policy_09, 0.8, optimal_policy_08)