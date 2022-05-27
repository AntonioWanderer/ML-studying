import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5*(1 + f(x))*(1 - f(x))

k = 5 #neurons in hidden layer
lx = 6 # input length

w1 = np.ones((k,lx)) / 2
w2 = np.ones(k) / 2

ln = []

def go_forward(inp):
    sum = w1 @ inp
    out = np.array([f(x) for x in sum])
    
    sum = w2 @ out
    y = f(sum)
    return (y, out)

def train(epoch):
    global w1, w2, ln, lx
    xl = lx
    lmd = 0.01
    n = 10000
    count = len(epoch)
    for k in range(n):
        x = epoch[np.random.randint(0, count)]
        y, out = go_forward(x[0:xl])
        e = y - x[-1]
        delta = e *df(y)
        w2 = w2 - lmd * delta * out
        
        delta2 = w2 * delta * df(out)
        for i in range(len(w2)):
            w1[i, :] = w1[i, :] - np.array(x[0:xl]) * delta2[i] * lmd
        
        ln.append(e)
        
#epoch = [(-1, -1, -1, -1),
#         (-1, -1, 1, 1),
#         (-1, 1, -1, -1),
#         (-1, 1, 1, 1),
#         (1, -1, -1, -1),
#         (1, -1, 1, 1),
#         (1, 1, -1, -1),
#         (1, 1, 1, -1)]

epoch = [(0.9, 0.8, 0.9, 0.4, 0.6, 0.7, 0.6),
         (0.4, 0.7, 0.3, 0.5, 0.2, 0.4, 0.2),
         (0.7, 0.8, 0.5, 0.2, 0.7, 0.7, 0.4),
         (0.8, 0.6, 0.6, 0.1, 0.2, 1.0, 1.0),
         (0.8, 0.8, 0.3, 0.4, 0.6, 0.5, 0.8)]

train(epoch)

for x in epoch:
    y, out = go_forward(x[0:lx])
    print(f"Value: {y} => {x[-1]}")
    
plt.plot(ln)
plt.show()