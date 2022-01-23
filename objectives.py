import numpy as np


def length(X):
    x1, x2 = X[0], X[2]
    y1, y2 = X[1], X[3]
    s = 1e-3
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * s

    return length


def cost(individ):
    cost = 0

    for ind in individ:
        number_of_points = len(ind)
        number_of_segments = int((number_of_points - 2) / 2)
        segments = []
        k = 0
        for _ in range(number_of_segments):
            j = k + 4
            segments.append(ind[k:j])
            k += 2

        cost += sum([length(segment) for segment in segments])

    return cost


def wave_h(hs):
    pass
