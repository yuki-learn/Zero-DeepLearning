import sys, os
import numpy as np
import pickle
sys.path.append(os.path.abspath('./src')) # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
from activation_function import sigmoid, softmax

# (訓練画像, 訓練ラベル), (テスト画像. テストラベル)という形式でデータを返す。
def get_data():
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
    pklファイルに保存された学習済みのパラメータを読み込む
    """
    filepath = os.path.join(os.path.dirname(__file__), "sample_weight.pkl")
    with open(filepath, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    ニューラルネットワークの推論処理
    手書き文字画像を1〜10に分類するので、出力層の数は"10個"
    
    x: 画像データ1枚(28x28)を1次元配列に変換しているので784個の要素があり、それらがすべて入力層となる。
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    
    # 入力画像を1枚ずつ推論処理に渡して、分類を行う。
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
        
        # 正解ラベルとニューラルネットワークの予測を比較
        if p == t[i]:
            accuracy_cnt += 1

    # 分類の結果、93.52% 正しく推論できたことを示す
    print(f"Accuracy: {str(float(accuracy_cnt) / len(x))}")