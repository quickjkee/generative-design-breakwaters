import numpy as np


def choose_best(solution, F1, F2):
    F = [(f1, f2, i) for i, (f1, f2) in enumerate(zip(F1, F2)) if f1 <= 0.4 and f2 <= 0.3]
    dist_idx = [(np.sqrt(f[0]**2 + f[1]**2), f[2]) for f in F]
    sorted_dist_idx = sorted(dist_idx)
    best_idx = sorted_dist_idx[0][1]

    return solution[best_idx], F1[best_idx], F2[best_idx]


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

        if number_of_segments == 1:
            if cost < 0.2:
                cost = cost * (np.log(1 / cost) + 1)

    return cost


def wave_h(hs):
    """
    hs - height of waves in every point of domain
    """

    return hs[23, 40]