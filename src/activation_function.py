import numpy as np

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


def softmax(x):
    """
    ソフトマックス関数
    """
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
