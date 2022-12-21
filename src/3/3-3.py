import numpy as np
from activation_function import sigmoid


def idnetity_function(x):
    return x

def init_param():
    param = {}
    param['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 入力層 -> 1層目の計算で使う重み 2x3
    param['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 1層目 -> 2層目の計算で使う重み 3x2
    param['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2層目 -> 出力層の計算で使う重み 2x2
    param['B1'] = np.array([0.1, 0.2, 0.3]) # 入力層 -> 1層目の計算で使うバイアス(1層目のニューロンの数だけ)
    param['B2'] = np.array([0.1, 0.2]) # 1層目 -> 2層目へのバイアス
    param['B3'] = np.array([0.1, 0.2]) # 2層目 -> 出力層へのバイアス
    
    return param
    
def forward(param, input):
    # 入力層 -> 1層目への信号伝達
    A1 = np.dot(input, param['W1']) + param['B1'] # 1層目のそれぞれの和(行列計算)
    Z1 = sigmoid(A1) # それぞれ活性化関数を通して1層目の出力とする。

    # 1層目 -> 2層目への信号伝達
    A2 = np.dot(Z1, param['W2']) + param['B2']
    Z2 = sigmoid(A2)

    # 2層目 -> 出力層への信号伝達
    A3 = np.dot(Z2, param['W3']) + param['B3']
    Y = idnetity_function(A3) # そのまま値を出力する「恒等関数」といい、出力層の活性化関数として使う。
    print(A3)
    print(Y)



if __name__ == '__main__':
    input = np.array([1.0, 0.5]) # 入力層
    param = init_param()
    forward(param, input)