import numpy as np


def cross_entropy_error(y, t):
    # データがバッチでないときの処理
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == '__main__':
    # 前回の手書き文字の教師データを例として、「2」を正解ラベルとする。
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # 推論した結果、「2」である確率が一番高い画像のニューラルネットワーク出力
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    
    # 推論した結果、「7」である確率が一番高い画像のニューラルネットワーク出力
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    
    # 損失関数に適用
    result1 = cross_entropy_error(np.array(y1), np.array(t))
    result2 = cross_entropy_error(np.array(y2), np.array(t))
    
    print(result1) # 0.510825457099338
    print(result2) # 2.302584092994546
    
    # 結果2より結果1のほうが損失関数の値が小さくなっている
    # -> 結果1のほうが教師データ「2」により適合していることを表している。
    