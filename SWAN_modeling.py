import subprocess
import numpy as np
from models import get_model, get_classifier_model
from visualization import plot_binary_mask
import constraints



PIX_TARGET = [127, 106]
THRESHOLD = 0.075
TARGET = constraints.TARGET
x = constraints.x
y = constraints.y
X, Y = np.meshgrid(x, y)


def modeling(population):
    hs_pop = []
    counter_swan = 0
    classifier = get_classifier_model()

    for individ in population:
        binary_mask = plot_binary_mask(individ)
        pred_for_individ = classifier.predict(binary_mask.reshape(1, 224, 224))
        print(pred_for_individ[0][0])


        if pred_for_individ[0][0] > THRESHOLD:
            counter_swan += 1
            _, hs_for_ind, _ = hs_modeling([individ])
            hs_pop.append(hs_for_ind[0])

        else:
            _, hs_for_surr, _ = surrogate_modeling_ind(binary_mask)
            hs_pop.append(hs_for_surr[0])

    return None, hs_pop, counter_swan


def hs_pred_func(pix):
  a = -0.45376437
  b = 0.53035618
  hs = a * pix + b
  return hs


def surrogate_modeling_ind(binary_mask):
    my_model = get_model()
    hs_target = []

    map_for_individ = my_model.predict(binary_mask.reshape(1, 224, 224))
    #pixel = map_for_individ.reshape(224, 224, 1)[PIX_TARGET[0], PIX_TARGET[1]]
    hs_target.append(map_for_individ[1][0][0])

    return None, hs_target, 0



def surrogate_modeling(population):
    my_model = get_model()
    hs_target = []
    maps = []
    predict_class = []
    classifier = get_classifier_model()

    for individ in population:
        binary_mask = plot_binary_mask(individ)
        predict_class.append(classifier.predict(binary_mask.reshape(1, 224, 224)))
        map_for_individ = my_model.predict(binary_mask.reshape(1, 224, 224))
        maps.append(map_for_individ[0])
        hs = map_for_individ[1][0][0]
        hs_target.append(hs)

    return hs_target, maps, predict_class


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