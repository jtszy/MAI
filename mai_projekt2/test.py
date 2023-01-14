import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tensorflow as tf
import random


x_test = np.load("x_data_test.npy")
y_test = np.load("y_data_test.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(600, input_dim = 400, activation = "sigmoid"),
    tf.keras.layers.Dense(400, activation="tanh"),
    tf.keras.layers.Dense(300, activation = "softplus"),
    tf.keras.layers.Dense(200, activation = "sigmoid"),
    tf.keras.layers.Dense(100, activation = "tanh"),
    tf.keras.layers.Dense(4, activation="softplus")
])

model.compile(optimizer = "adamax", loss ="MSE")

model.load_weights("model1_weights.h5")

model.evaluate(x_test,y_test,verbose = 1, batch_size=1)

y_predict = model.predict(x_test, batch_size=1)


def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a * x - b * x * y, c * x * y - d * y]

def random_example():

    index = random.randint(0,199)

    x = x_test[index]
    x = x.reshape((2,200))
    t1 = np.linspace(40, 50, 200)
    t2 = np.linspace(50, 60, 200)

    solution = solve_ivp(lotkavolterra,
                        [50, 60], # przedział czasowy
                        [x[0][199],x[1][199]], # punkty startowe
                        args=(y_predict[index]), #parametry równania Lotki-Voltery
                        dense_output=True)

    z = solution.sol(t2)


    plt.plot(t1, x[0])
    plt.plot(t1, x[1])
    plt.plot(t2,z.T, linestyle = "dashed")
    plt.xlabel('Czas')
    plt.legend(['Prey', 'Predator','predicted Prey', 'predicted Predator'], shadow=True)

    plt.title('Równanie Lotki-Voltery')
    plt.show()

for i in range(10):
    print(i)
    random_example()