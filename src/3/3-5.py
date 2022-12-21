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
    batch_size = 100 # まとめて推論する画像の数
    accuracy_cnt = 0
    
    # 入力画像を100枚ずつ取り出す。
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        
        # 100枚まとめて渡す
        y_batch = predict(network, x_batch)
        
        # 各推論結果の最大値のインデックスを100枚分返す。
        p = np.argmax(y_batch, axis=1)
        
        # 100枚の中で合っている数をカウント
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print(f"Accuracy: {str(float(accuracy_cnt) / len(x))}")