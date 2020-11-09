# Simulated Annealing

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import random

interval = (-3, 3)

def simulated_annealing(random_start, cost_function, num_variables, random_neighbour, acceptance, temperature, maxsteps=1000):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state_X = random_start()
    state_Y = random_start()
    cost = 0
    states = [[], [], []]
    costs = []

    if num_variables == 2:
      cost = cost_function(state_X, state_Y)
    else:
      cost = cost_function(state_X)
    
    states[0] = [state_X]
    states[1] = [state_Y]
    states[2] = [cost]

    
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_state_X = random_neighbour(state_X, fraction)
        new_state_Y = random_neighbour(state_Y, fraction)
        new_cost = []

        if num_variables == 2:
          new_cost = cost_function(new_state_X, new_state_Y)
        else:
          new_cost = cost_function(new_state_X)
  
        if acceptance_probability(cost, new_cost, T) > rn.random():
            state_X, state_Y, cost = new_state_X, new_state_Y, new_cost
            states[0].append(state_X)
            states[1].append(state_Y)
            states[2].append(cost)
            costs.append(cost)
    
    return [state_X, state_Y], costs[-1], states, costs

def clip(x):
    """ Force x to be in the interval."""
    return max(min(x, interval[1]), interval[0])

def random_start():
    """ Random point in the interval."""
    return interval[0] + (interval[1] - interval[0]) * rn.random_sample()

def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    delta = (-amplitude/2.) + amplitude * rn.random_sample()
    return clip(x + delta)

def acceptance_probability(cost, new_cost, temperature):
    """Calculates the probability of accepting the state."""
    if new_cost < cost:
        return 1
    else:
        return np.exp(- (new_cost - cost) / temperature)

def temperature(fraction):
    """Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

def plot_annealing(states, costs, cost_function, num_variables):
    """Plots 1 variable or 2 variable functions, the states and costs."""
    if num_variables == 2:
        fig = plt.figure(figsize = (20, 10))
        fig.suptitle("Evolution of states and costs of the simulated annealing")
        ax1 = fig.add_subplot(121, projection='3d')
        x = y = np.arange(interval[0], interval[1], 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(cost_function(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)
        ax1.plot_surface(X, Y, Z, alpha = 0.3, cmap='hot')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x, y)')
        ax1.set_title("States")
        ax1.scatter(states[0], states[1], states[2], marker = "o")
        ax2 = fig.add_subplot(122)
        ax2.plot(costs, 'b')
        ax2.set_title("Costs")

    elif num_variables == 1: 
        plt.figure(figsize = (20, 10))
        plt.suptitle("Evolution of states and costs of the simulated annealing")
        plt.subplot(121)
        plt.plot(states[0], 'r')
        plt.title("States")
        plt.subplot(122)
        plt.plot(costs, 'b')
        plt.title("Costs")
        plt.show()

def visualize_annealing(cost_function, num_variables):
    """A one liner function to call Simulated Annealing on the passed function, log the result and plot the graphs."""

    state, c, states, costs = simulated_annealing(random_start, cost_function, num_variables, random_neighbour, acceptance_probability, temperature, maxsteps=1000)
    if num_variables == 2:
      print("Global Minima at x = {0}, y = {1}, for which, the value of f(x, y) = {2}.".format(state[0], state[1], c))
    elif num_variables == 1:
      print("Global Minima at x = {0}, for which, the value of f(x) = {1}.".format(state[0], c))

    plot_annealing(states, costs, cost_function, num_variables)
    return state, c

if __name__=="__main__":
    visualize_annealing(lambda x: x**2, 1)
    visualize_annealing(lambda x, y: x**2 + y**2, 2)
    visualize_annealing(lambda x, y: (x**2 - y**2) * np.sin(x + y) / (x**2 + y**2), 2)