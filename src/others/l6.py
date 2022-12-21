# 4.4から

import numpy as np

from util import cross_entropy_error, numerical_gradient, softmax


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        """
        予測するためのメソッド
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        損失関数の値を求める
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

def function_2(x):
    return x[0]**2 + x[1]**2


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x

def f(W):
    return net.loss(x, t)

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))

# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))

dW = numerical_gradient(f, net.W)
print(dW)