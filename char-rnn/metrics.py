
import math


def mean(array):
    """
    calculate array mean
    """
    res = sum(array) / len(array) if len(array) > 0 else 0
    return res


def cal_perplexity(loss):
    """
    计算训练和验证时的困惑度
    """
    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
    return perplexity