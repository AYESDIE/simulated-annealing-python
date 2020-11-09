import numpy as np
import matplotlib.pyplot as plt

def h(x):
    if x < -1 or x > 1:
        y = 0
    else:
        y = (np.cos(50 * x) + np.sin(20 * x))
    return y

hv = np.vectorize(h)

X = np.linspace(-1, 2, num = 1000)
plt.plot(X, hv(X))
plt.savefig("graph.png")
plt.close()

def SA(search_space, func, T):
    scale = np.sqrt(T)
    start = np.random.choice(search_space)
    x = start * 1
    cur = func(x)
    history = [x]
    for i in range(1000):
        prop = x + np.random.normal() * scale 
        if prop > 1 or prop < 0 or np.log(np.random.rand()) * T > (func(prop) - cur):
            prop = x
        x = prop
        cur = func(x)
        T = 0.9 * T
        history.append(x)
    return x, history

X = np.linspace(-1, 1, num = 1000)
x1, history = SA(X, h, T = 4)
plt.plot(X, hv(X))
plt.scatter(x1, hv(x1), marker = 'x')
plt.plot(history, hv(history))
plt.savefig("ple.png")

print(h(history[-1]))