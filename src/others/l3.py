from this import d
from turtle import shape
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

# 全ての計算
def forword(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W2']
    b1, b2, b3 = network['b1'], network['b2'], network['b2']
    
    a1 = np.dot(x, W1) + b1 # 1層目の計算
    z1 = sigmoid(a1) # 1層目を活性化関数に適用

    a2 = np.dot(z1, W2) + b2 # 2層目の計算
    z2 = sigmoid(a2) # 2層目を活性化関数に適用

    a3 = np.dot(z2, W3) + b3 # 3層目の計算
    # 最後の活性化関数は恒等関数を使用する。
    return identity_function(a3)


# 0次元目の要素が3, 1次元目の要素が2
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)


# 1次元目の要素が2
B = np.array([7, 8])
print(B.shape)

print(np.dot(A, B))


# 3.3.3 
X = np.array([1, 2])
W = np.array([
    [1, 3, 5], 
    [2, 4, 6]
    ])

# (1*1)+(2*2) = 5
# (1*3)+(2*4) = 11
# (1*5)+(2*6) = 17

print(W)
Y = np.dot(X, W)
print(Y)


print("----------------")
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)
print("----------------")

# 1層目を活性化関数に適用する。
Z1 = sigmoid(A1)

print(A1) # [0.3, 0.7, 1.1]
print(Z1) # [0.57444252, 0.66818777, 0.75026011]

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2]) # 

print(Z1.shape)
print(W2.shape)
print(B2.shape)

# 1層目 -> 2層目の計算
A2 = np.dot(Z1, W2) + B2

# 2層目の結果に活性化関数を適用
Z2 = sigmoid(A2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)