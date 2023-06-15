# import numpy as np
# import matplotlib.pyplot as plt

# def y_hat(x):
#     return x ** 2

# def  y_derivative(x):
#     return 2 * x

# x = np.linspace(-0.5, 0.2, 0.01)
# y = x **2

# starting_position = (0, y_hat(0))
# learning_rate = 0.01

# for _ in range(5):
#     new_x = starting_position[0] - learning_rate * y_derivative(starting_position[0])
#     new_y = y_hat(new_x)
#     plt.scatter(new_x, new_y, color='red')
#     plt.plot(x, y)
#     # plt.show()
#     plt.pause(0.01)
#     plt.clf()

import numpy as np
import matplotlib.pyplot as plt

def y_hat(x, theta1, theta2):
    return x * theta1 + theta2

theta1 = -0.5
theta2 = 0.2
learning_rate = 0.01
max_iter = 5


x = np.arange(0, 1, 0.01)
y = x + np.random.normal(0, 0.2, len(x))

for i in range(max_iter):
    gradienttheta1 = np.mean((y_hat(x, theta1, theta2) - y) * x)
    gradienttheta2 = np.mean(y_hat(x, theta1, theta2) - y)

    theta1 = theta1 - learning_rate * gradienttheta1
    theta2 = theta2 - learning_rate * gradienttheta2
prediction = y_hat(x, theta1, theta2)
plt.scatter(x, y, color='red')
plt.plot(x,prediction, color='blue')
plt.show()