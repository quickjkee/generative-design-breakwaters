import numpy as np
from numpy.linalg import norm as euclid_norm

"EVERYTHING ABOUT CONSIDERED DOMAIN"

TARGET = [23, 40]
x = np.linspace(0, 2075, 84)
y = np.linspace(0, 1450, 59)
X, Y = np.meshgrid(x, y)
Z = (X + 25) * (Y + 10) / (32 * (X + Y + 50))
V1_PROH_AREA = [578, 900, 1480, 354]
V2_PROH_AREA = [578, 800, 1480, 200]


def vector_mult(vector1, vector2):
    biased_x_1 = vector1[2] - vector1[0]
    biased_y_1 = vector1[3] - vector1[1]

    biased_x_2 = vector2[2] - vector2[0]
    biased_y_2 = vector2[3] - vector2[1]

    return biased_x_1 * biased_y_2 - biased_y_1 * biased_x_2


def nessecary_cond(vector1, vector2):
    # check for intersection of proection on x and y axis
    # vector = [x1, y2, x2, y2]
    min_x_v1, max_x_v1 = min(vector1[0], vector1[2]), max(vector1[0], vector1[2])
    min_y_v1, max_y_v1 = min(vector1[1], vector1[3]), max(vector1[1], vector1[3])
    min_x_v2, max_x_v2 = min(vector2[0], vector2[2]), max(vector2[0], vector2[2])
    min_y_v2, max_y_v2 = min(vector2[1], vector2[3]), max(vector2[1], vector2[3])

    return max(min_x_v1, min_x_v2) <= min(max_x_v1, max_x_v2) and max(min_y_v1, min_y_v2) <= min(max_y_v1, max_y_v2)


def sufficient_cond(vector1, vector2):
    fixed = 1.2e+4
    first = vector1
    one_first = [first[0], first[1], vector2[0], vector2[1]]
    two_first = [first[0], first[1], vector2[2], vector2[3]]

    second = vector2
    one_second = [second[0], second[1], vector1[0], vector1[1]]
    two_second = [second[0], second[1], vector1[2], vector1[3]]

    d1 = vector_mult(first, one_first)
    d2 = vector_mult(first, two_first)

    d3 = vector_mult(second, one_second)
    d4 = vector_mult(second, two_second)

    if ((d1 <= 0 and d2 >= 0) or (d1 >= 0 and d2 <= 0)) and ((d3 <= 0 and d4 >= 0) or (d3 >= 0 and d4 <= 0)):
        if ((d1 == 0 and d2 != 0) or (d1 != 0 and d2 == 0)) and ((d3 == 0 and d4 != 0) or (d3 != 0 and d4 == 0)):
            if abs(d1) <= fixed and abs(d2) <= fixed and abs(d3) <= fixed and abs(d4) <= fixed:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def intersection_segments(points):
    if len(points) == 4:
        return 0

    vectors = []
    num_of_vectors = int(len(points) / 2) - 1

    first = 0
    second = 2
    for _ in range(num_of_vectors):
        vectors.append(points[first:second] + points[second:second + 2])
        first += 2
        second += 2

    for i, vector1 in enumerate(vectors):
        for j in range(i + 1, num_of_vectors):
            if not nessecary_cond(vector1,vectors[j]):  # if nessecary cond is not success then vectors are not intersection
                continue
            else:
                if sufficient_cond(vector1, vectors[j]):
                    return 1
                else:
                    continue

    return 0


def distance_to_vector(vector):
    target = TARGET
    ts_X = [X[target[0], target[1]]]
    ts_Y = [Y[target[0], target[1]]]
    dist = []

    for x, y in zip(ts_X, ts_Y):
        v = [vector[2] - vector[0], vector[3] - vector[1]]
        w0 = [x - vector[0], y - vector[1]]
        w1 = [x - vector[2], y - vector[3]]

        w0_v = np.dot(w0, v)
        w1_v = np.dot(w1, v)
        if w0_v < 0:
            dist.append(euclid_norm(np.array([x, y]) - np.array([vector[0], vector[1]])))
        elif w1_v > 0:
            dist.append(euclid_norm(np.array([x, y]) - np.array([vector[2], vector[3]])))
        else:
            x1 = v[0]
            y1 = v[1]
            x2 = w0[0]
            y2 = w0[1]
            mod = np.sqrt(x1 * x1 + y1 * y1)
            dist.append(abs(x1 * y2 - y1 * x2) / mod)

    return dist


def point_on_curve(points):
    target = TARGET
    vectors = []
    num_of_vectors = int(len(points) / 2) - 1
    fixed = 60

    first = 0
    second = 2
    for _ in range(num_of_vectors):
        vectors.append(points[first:second] + points[second:second + 2])
        first += 2
        second += 2

    for vector in vectors:
        dist = distance_to_vector(vector)
        check_constr = list(d <= fixed for d in dist)
        if any(check_constr):
            return 1

    return 0


def distance_between_vectors(points):
    vectors = []
    num_of_vectors = int(len(points) / 2) - 1
    fixed = 50

    first = 0
    second = 2
    for _ in range(num_of_vectors):
        vectors.append(points[first:second] + points[second:second + 2])
        first += 2
        second += 2

    dists = []
    for i, vector1 in enumerate(vectors):
        for j in range(i + 1, num_of_vectors):
            vector_sum = vector1[0:2] + vectors[j][2:4]
            x = np.array(vector_sum[2] - vector_sum[0])
            y = np.array(vector_sum[3] - vector_sum[1])
            norm = np.sqrt(x ** 2 + y ** 2)
            if norm <= fixed:
                return 1
            break

    return 0


def bw_length(points):
    ls = []
    num_of_vectors = int(len(points) / 2) - 1
    fixed = 90

    first = 0
    second = 2
    for _ in range(num_of_vectors):
        vector = points[first:second] + points[second:second + 2]
        x = vector[2] - vector[0]
        y = vector[3] - vector[1]
        L = np.sqrt(x ** 2 + y ** 2)
        if L <= fixed:
            return 1

        first += 2
        second += 2

    return 0

def intersection_breakwaters(individ):
    vectors_individ = []

    for points in individ:
        vectors = []
        first = 0
        second = 2
        num_of_vectors = int(len(points) / 2) - 1
        for _ in range(num_of_vectors):
            vectors.append(points[first:second] + points[second:second + 2])
            first += 2
            second += 2
        vectors_individ += vectors

    for i, vector1 in enumerate(vectors_individ):
        for vector2 in vectors_individ[i + 1:]:
            if not nessecary_cond(vector1,
                                  vector2):  # if nessecary cond is not success then vectors are not intersection
                continue
            else:
                if sufficient_cond(vector1, vector2):
                    return 1
                else:
                    continue

    return 0


def domain_limits(points):
    V = [V1_PROH_AREA, V2_PROH_AREA]
    domain = [[570, 1480],  # x-axis
              [200, 1100]]
    X = points[::2]
    Y = points[1:][::2]

    for x, y in zip(X, Y):
        if x < domain[0][0] or x > domain[0][1]:
            return 1
        elif y < domain[1][0] or y > domain[1][1]:
            return 1

    vectors = []
    first = 0
    second = 2
    num_of_vectors = int(len(points) / 2) - 1
    for _ in range(num_of_vectors):
        vectors.append(points[first:second] + points[second:second + 2])
        first += 2
        second += 2

    for v_domain in V:
        for v_individ in vectors:
            if not nessecary_cond(v_domain,
                                  v_individ):  # if nessecary cond is not success then vectors are not intersection
                continue
            else:
                if sufficient_cond(v_domain, v_individ):
                    return 1
                else:
                    continue

    return 0


def check_constraints(individ):
    constr = []
    if len(individ) > 1:
        for points in individ:
            constr.append(bw_length(points) or distance_between_vectors(points) or point_on_curve(points) or intersection_segments(points) or domain_limits(points))
        constr.append(intersection_breakwaters(individ))

    elif len(individ) == 1:
        constr.append(bw_length(individ[0]) or distance_between_vectors(individ[0]) or point_on_curve(individ[0]) or intersection_segments(individ[0]) or domain_limits(individ[0]))

    return any(constr)