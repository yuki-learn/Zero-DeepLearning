
# 4.2.1 2乗和誤差から


import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from util import mean_squared_error, numerical_diff, numerical_gradient

np.set_printoptions(precision=12)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2




def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x
        
        
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0

print(numerical_diff(function_tmp1, 3.0))

# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# r = mean_squared_error(np.array(y), np.array(t))
# print(r)

# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# r = mean_squared_error(np.array(y), np.array(t))
# print(r)



# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)
# print(t_train.shape)

# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# print(x_batch)
# print(t_batch)


# print("---------------")

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# # plt.show()

# print("---------------")
# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
