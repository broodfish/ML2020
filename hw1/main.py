import numpy as np
import matplotlib.pyplot as plt 
import math
import csv
import os

def readfile(filename): # reading the data points in file
    x = []
    y = []
    with open(filename) as file:
        read = csv.reader(file, delimiter=',')
        for row in read:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

def show_fitting_line(x): # print out the fitting line function
    n = len(x) - 1
    print('Fitting line: ',end='')
    #x^(n-1) ~ x^1
    for i in range(n):
        print(x[i][0], 'X ^', n - i, ' + ', end='')
    #x^0
    print(x[n][0])

def plot(xi, yi, lse_x, nt_x): # plot data points and fitting lines
    xi_min = min(xi)
    xi_max = max(xi)
    x = np.linspace(xi_min - 1, xi_max + 1, 500)
    # rLSE
    plt.subplot(2, 1, 1)
    plt.title('LSE')
    plt.plot(xi, yi, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(lse_x)):
        y += lse_x[i] * np.power(x, len(lse_x) - 1 - i)
    plt.plot(x, y, '-k')
    # Newton's method
    plt.subplot(2, 1, 2)
    plt.title('Newton\'s Method')
    plt.plot(xi, yi, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(nt_x)):
        y += nt_x[i] * np.power(x, len(nt_x) - 1 - i)
    plt.plot(x, y, '-k')
    plt.tight_layout(pad=0.4, w_pad=4.0, h_pad=3.0)
    plt.show()

def T(x): # transpose of matrix x
    transpose = np.zeros((x.shape[1], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            transpose[j][i] = x[i][j]
    return transpose

def ADD(A, B): # A + B
    m = max(A.shape[0], B.shape[0])
    n = max(A.shape[1], B.shape[1])
    add = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if (i < A.shape[0]) & (j < A.shape[1]):
                a = A[i][j]
            else:
                a = 0
            if (i < B.shape[0]) & (j < B.shape[1]):
                b = B[i][j]
            else:
                b = 0
            add[i][j] = a + b
    return add

def MUL(A, B): # AB
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[0]
    k = B.shape[1]
    mul = np.zeros((m, k))
    if (n != p):
        print("MUL error!")
        return
    for i in range(m):
        for j in range(n):
            for t in range(k):
                mul[i][t] += A[i][j] * B[j][t]
    return mul

def MINUS(A, B): # A - B
    minus = ADD(A, MUL_S(-1, B))
    return minus

def I(n): # build a nxn identity matrix
    I = np.zeros((n, n))
    for i in range(n):
        I[i][i] = 1
    return I

def MUL_S(s, x): # multiply matrix by a scalar
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = s * x[i][j]
    return x

def INV(x): # the inverse matrix of x
    m = x.shape[0]
    n = x.shape[1]
    if (m != n):
        print("INV error!")
        return
    inverse = I(m)
    for i in range(m):
        t1 = x[i][i]
        for j in range(m):
            x[i][j] = x[i][j] / t1
            inverse[i][j] = inverse[i][j] / t1
        for k in range(i + 1, m):
            t2 = -x[k][i]
            for j in range(m):
                x[k][j] += t2 * x[i][j]
                inverse[k][j] += t2 * inverse[i][j]
    
    for i in range(m - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            t3 = -x[k][i]
            for j in range(m):
                x[k][j] += t3 * x[i][j]
                inverse[k][j] += t3 * inverse[i][j]
    return inverse

def get_loss(A, x, b): # calculate the loss of fitting line
    loss = np.sum(np.square(ADD(MUL(A, x), MUL_S(-1, b)))) # A@x - b
    return loss

def rLSE(A, b, lse_lambda): # LSE method
    n = A.shape[1]
    AT = T(A)
    # np.linalg.inv(A.T@A + lse_lambda*np.identity(n))@A.T@b
    x = MUL(MUL(INV(ADD(MUL(AT, A), MUL_S(lse_lambda, I(n)))), AT), b)
    loss = get_loss(A, x, b)
    return x, loss

def newton(A, b): # Newton' s method
    n = A.shape[1]
    x0 = np.zeros((n,1))
    for i in range(n):
        x0[i] = 100
    eps = 100
    while eps > 1e-6:
        AT2 = MUL_S(2, T(A))
        AT2A = MUL(AT2, A)
        AT2b = MUL(AT2, b)
        AT2Ax = MUL(AT2A, x0)
        # (np.linalg.inv(2*A.T@A))@(2*A.T@A@x0 - 2*A.T@b)
        x1 = x0 - MUL(INV(AT2A), MINUS(AT2Ax, AT2b))
        eps = abs(np.sum(np.square(MINUS(x1, x0)))/n) # x1 - x0
        x0 = x1
    loss = get_loss(A, x0, b)
    return x0, loss

filename = input('filename: ') # the filename consisting of data points
pbases = int(input('the number of polynomial bases n: ')) # the number of polynomial bases n
lse_lambda = float(input('lambda for LSE case: ')) # lambda for LSE case

xi, yi = readfile(filename)
A = np.zeros((len(xi), pbases))
for j in range(pbases):
    A[:, j] = np.power(xi, pbases - 1 - j).reshape(-1)
b = np.asarray(yi, dtype='float').reshape((-1, 1))

# LSE
lse_x, lse_loss = rLSE(A, b , lse_lambda)
print('\nLSE:')
show_fitting_line(lse_x)
print('Total error: ', lse_loss)

# Newton's method
b = np.asarray(yi, dtype='float').reshape((-1, 1))
nt_x, nt_loss = newton(A, b)
print('\nNewton\'s Method:')
show_fitting_line(nt_x)
print('Total error: ', nt_loss)

# draw the graph
plot(xi, yi, lse_x, nt_x)