import numpy as np
from util import cross_entropy_error, numerical_gradient, softmax
np.set_printoptions(precision=12)

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# t = np.array([0, 0, 1])

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

# f = lambda w: net.loss(x, t)
# dW = numerical_gradient(f, net.W)

# print(dW)
