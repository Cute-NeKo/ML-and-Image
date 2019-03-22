import numpy as np
import matplotlib.pyplot as plt


def perceptron(data, label, step, a):
    w = np.random.random(data.shape[1] + 1) - 0.5
    data = np.insert(data, 0, np.ones(data.shape[0]), 1)

    for i in range(step):
        for j in range(data.shape[0]):
            x = data[j, :]
            y = label[j]
            if (np.dot(x, w) * y < 0):
                w += a * y * x
    return w


data = np.array([[3, 3], [4, 3], [1, 1], [1, 2], [2, 2]])
label = np.array([-1, -1, 1, 1, 1])

#二次的拟合
data2 = data ** 2
w = perceptron(data2, label, 10000, 0.02)

print(w)
print(np.dot(data, w[1:]) + w[0])
x = np.linspace(-10, 10, 100)
y = np.sqrt((w[0] + w[1] * (x ** 2)) / -w[2])

plt.plot(x, y)
plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()
