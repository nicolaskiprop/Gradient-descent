import numpy as np
def basic_gradient_descent(gradient, start, learning_rate, iterations=100, tolerance=1e-6):
    vector = start
    for _ in range(iterations):
        diff = -learning_rate * gradient(vector)
        if np.all(np.abs(diff) < tolerance):
            break
        vector += diff
    return vector
