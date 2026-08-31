import numpy as np


def cumsum0(x, axis=None):
    return np.insert(np.cumsum(x, axis=axis), 0, 0.0, axis=axis)


def ER(x):
    dx = np.diff(x)
    ER = dx[1:] / dx[:-1]
    ER[ER < 1.0] = 1.0 / ER[ER < 1.0]
    return ER
