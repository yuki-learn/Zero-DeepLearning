# 3章の3.5から
import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from util import softmax, sigmoid


def get_data():
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
       network = pickle.load(f)
    
    return network
    
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
    
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)



x, t = get_data()
network = init_network()

batch_size = 100

accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 最も確率の高い要素のインデックスを取得
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print(f"Accuracy: {float(accuracy_cnt) / len(x)}")