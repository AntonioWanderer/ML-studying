import numpy as np

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5*(1 + f(x))*(1 - f(x))

def direct(w1, w2, inp):
    z = w1 @ inp
    out1 = f(z)
    z = w2 @ out1
    out = f(z)
    return(out1, out)
    
def train(w1, w2, inp, outp, lam):
    out1, out = direct(w1, w2, inp)
    e = out - outp
    lgrad = e * df(out)
    w2 = w2 - lam * lgrad * out
    lgrad2 = w2 * lgrad * df(out1)
    for i in range(len(w2)):
        w1[i,:] = w1[i,:] - lam * lgrad2[i] * np.array(inp[0:len(inp)])
    return(w1, w2)

epoch = [(-1, -1, -1),
         (-1, -1, 1),
         (-1, 1, -1),
         (-1, 1, 1),
         (1, -1, -1),
         (1, -1, 1),
         (1, 1, -1),
         (1, 1, 1)]

res = [-1, 1, -1, 1, -1, 1, -1, -1]

hid_len = 5
w1 = np.ones((hid_len,3)) / 2
w2 = np.ones(hid_len) / 2

lam = 0.01
k = 10000
for i in range(k):
    n = np.random.randint(len(res))
    w1, w2 = train(w1, w2, epoch[n], res[n], lam)
err = 0
for i in range (len(res)):
    y1, y = direct(w1, w2, epoch[i])
    err = err + ((y - res[i]) / 2) ** 2
    print(y, res[i])
err = err / len(res)
print("Err", err)