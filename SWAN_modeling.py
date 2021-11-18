import subprocess
import numpy as np
from models import get_model
from visualization import plot_map, plot_binary_mask
import constraints


PIX_TARGET = [127, 106]
TARGET = constraints.TARGET
x = constraints.x
y = constraints.y
X, Y = np.meshgrid(x, y)


def modeling(population):
    first_cond = 200
    second_cond = 260
    hs_pop = []
    counter_swan = 0

    for individ in population:
        vectors = []
        dist = []
        num_of_vectors = int(len(individ) / 2) - 1

        first = 0
        second = 2
        for _ in range(num_of_vectors):
            vectors.append(individ[first:second] + individ[second:second + 2])
            first += 2
            second += 2

        for vector in vectors:
            dist.append(constraints.distance_to_vector(vector, TARGET))

        min_dist = min(dist)[0]
        if min_dist < first_cond or min_dist > second_cond:
            hs_pop.append(surrogate_modeling([individ])[0])
        else:
            counter_swan += 1
            hs_for_ind, _ = hs_modeling([individ])
            hs_pop.append(hs_for_ind[0])

    return hs_pop, counter_swan


def hs_pred_func(pix):
  a = -0.45376437
  b = 0.53035618
  hs = a * pix + b
  return hs


def surrogate_modeling(population):
    my_model = get_model()
    hs_target = []
    maps = []

    for individ in population:
        binary_mask = plot_binary_mask(individ)
        map_for_individ = my_model.predict(binary_mask.reshape(1, 224, 224))
        maps.append(map_for_individ)
        pixel = map_for_individ.reshape(224, 224, 1)[PIX_TARGET[0], PIX_TARGET[1]]
        hs_target.append(hs_pred_func(pixel)[0])

    return hs_target, maps


def hs_modeling(population):
    path_to_input = 'INPUT'
    path_to_hs = 'r/hs47dd8b1c0d4447478fec6f956c7e32d9.d'

    hs_target = []
    Z = []

    for individ in population:
        file_to_read = open(path_to_input, 'r')
        content_read = file_to_read.read()

        individs_for_input = []
        for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
        num_of_bw = len(individ)
        for j, ind in enumerate(individ):
            num_of_points = len(ind)
            for i, gen in enumerate(ind):
                if (i + 1) % 2 == 0:
                    if (i + 1) == num_of_points:
                        for_input += str(1450 - gen)
                    else:
                        for_input += str(1450 - gen) + ', '
                else:
                    for_input += str(gen) + ', '

            if j == (num_of_bw - 1):
                for_input += '\n$optline'
            else:
                for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '

        content_to_replace = for_input
        content_write = content_read.replace(content_read[content_read.find('\nOBSTACLE'):content_read.rfind('\n$optline') + 9], content_to_replace)
        file_to_read.close()

        file_to_write = open(path_to_input, 'w')
        file_to_write.writelines(content_write)
        file_to_write.close()

        subprocess.call('swan.exe')
        hs = np.loadtxt(path_to_hs)

        Z.append(hs)
        hs_target.append(hs[TARGET[0], TARGET[1]])

    return Z, hs_target, len(population)