import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    """
    シグモイド関数
    """
    return 1 / (1 + np.exp(-x))


def step_function(x):
    """
    ステップ関数
    """
    return np.array(x > 0, dtype=np.int64)

def relu_function(x):
    """
    ReLU関数
    """
    return np.maximum(0, x)


def plot_step_function(x):
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y 軸の範囲を指定
    plt.show()

def plot_sigmoid(x):
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y 軸の範囲を指定
    plt.show()
    
if __name__ == "__main__":
    # arange: -5.0 から 5.0まで0.1刻みの配列
    x = np.arange(-5.0, 5.0, 0.1)
    # plot_step_function(x)
    
    x = np.arange(-5.0, 5.0, 0.1)
    # plot_sigmoid(x)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu_function(x)
    plt.plot(x, y1, label='step')
    plt.plot(x, y2, label='sigmoid')
    plt.plot(x, y3, label='ReLU')
    plt.ylim(-0.1, 1.1) # y 軸の範囲を指定
    plt.legend(loc="upper left", fontsize=10)
    plt.show()
    
    
    
