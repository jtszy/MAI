import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# import tensorflow as tf
import random

# ZDEFINIOWANIE PRZEDZIAŁU ROZWĄZANIA
start = 0
end = 50
partition = 1000


# ROZWIAZYWANIE ROWNANIA ROZNICZKOWEGO

def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, c*x*y - d*y]

def preparing_lotka_data(a,b,c,d):
    solution = solve_ivp(lotkavolterra,
                    [start, end], # przedział czasowy
                    [random.randint(1,20),random.randint(1,20)], # punkty startowe
                    args=(a, b, c, d), #parametry równania Lotki-Voltery
                    dense_output=True)

    t = np.linspace(start, end, partition)
    z = solution.sol(t)

    x = [z.T[800:1000]]

    res_x = [0]*400

    for i in range(200):
        res_x[i] = x[0][i][0]
        res_x[200+i] = x[0][i][1]

    res_y = [a,b,c,d]

    return res_x, res_y

# STWORZENIE ZBIORU DANYCH UCZĄCYCH

x = []
y = []

samples_num = 200 #liczba przykladow

for i in range(samples_num):
    if i % 100 == 0:
        print(i)

    random_coefficients = 3*np.random.rand(4) + 1
    a,b,c,d = random_coefficients

    res_x, res_y = preparing_lotka_data(a,b,c,d)
    x += [res_x]
    y += [res_y]

x = np.asarray(x,float)
y = np.asarray(y,float)

np.save("x_data_test",x)
np.save("y_data_test",y)