from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

prec = 10
l = 50
inp = np.array([[1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 2, 0],
     [2, 1, 5],
     [0, 4, 0]])
out = np.array([1, 0, 2, 1, 5, 3])

def er(w1, w2, w3):
    global inp
    global out
    e = 0
    w3 = (w3 - prec / 2) * 2 * l / prec
    for t in range(len(out)):
        e = e + ((inp[t][0]*w1 + inp[t][1]*w2 + inp[t][2]*w3) - out[t]) ** 2
    return(e)


x = np.linspace(-l, l, prec) 
y = np.linspace(-l, l, prec)
X, Y = np.meshgrid(x, y)
ax = plt.axes(projection ='3d')
for w3 in range(prec):
    Z = er(X, Y, w3) 
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis')

plt.show()