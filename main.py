import numpy as np
import constraints
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import os

from visualization import plot_map, plot_binary_mask
from models import get_model
from spea2 import SPEA2_optimizer
from SWAN_modeling import hs_modeling, surrogate_modeling, modeling
from funcs import cost, wave_h, choose_best



if __name__ == '__main__':

    PIX_TARGET = [127, 106]

    TARGET = constraints.TARGET
    x = constraints.x
    y = constraints.y
    X, Y = np.meshgrid(x, y)
    Z = constraints.Z

    func_to_optimize = [cost, wave_h]

    spea2 = SPEA2_optimizer(pop_size=30,
                            arch_size=20,
                            max_iter=20,
                            mutat_rate_pop=0.35,
                            mutat_rate_individ=1.0,
                            domain=[[570, 1480],  # x-axis
                                    [200, 1100]],  # y-axis
                            )

    solution, history, history_hs_arch = spea2.optimize(func_to_optimize, hs_modeling)
    f1 = [cost(individ) for individ in solution]
    f2 = spea2.hs_arch

    matplotlib.use('tkagg')
    plt.scatter(f1, f2)
    plt.show()

    f1_hist = [[cost(individ) for individ in pop] for pop in history]
    f2_hist = history_hs_arch

    for i in range(len(f1_hist)):
        plt.scatter(f1_hist[i], f2_hist[i], label = str(i))
    plt.legend()
    plt.show()

    Z_sol = hs_modeling(solution)[0]
    for i in range(len(Z_sol)):
        plot_map(X, Y, Z_sol[i], solution[i])


    """
    result_for_types = []
    SWAN_for_types = []

    for type_modeling in [modeling, hs_modeling]:
        spea2 = SPEA2_optimizer(pop_size=20,
                                arch_size=15,
                                max_iter=20,
                                target=TARGET,
                                mutat_rate_pop=0.2,
                                mutat_rate_individ=1.0,
                                domain=[[570, 1480],  # x-axis
                                        [200, 1100]],  # y-axis
                                )

        solution, history, history_fit_arch, history_hs_arch, counter_SWAN = spea2.optimize(func_to_optimize, type_modeling)

        f1 = [cost(individ) for individ in solution]
        f2 = spea2.hs_arch

        F1 = []
        F2 = []
        for sol, hs_pop in zip(history, history_hs_arch):
            f = [cost(individ) for individ in sol]
            F1.append(f)
            f_ = [hs for hs in hs_pop]
            F2.append(f_)

        integrals = [rectangle_integral(f1, f2) for f1, f2 in zip(F1, F2)]
        result_for_types.append(integrals)
        SWAN_for_types.append(counter_SWAN)

    #mean_for_integrals = [np.sum((result_for_types[::2]), axis=0) / 5, np.sum((result_for_types[1:][::2]), axis=0) / 5]
    #mean_for_swan = [np.sum(SWAN_for_types[::2], axis=0) / 5, np.sum(SWAN_for_types[1:][::2], axis=0) / 5]

    matplotlib.use('tkagg')
    type = ['CNN + SWAN, ', 'SWAN, ']
    i = 0
    for result, counter_SWAN in zip(result_for_types, SWAN_for_types):
        plt.plot(result, label = type[i] + 'SWAN calc = ' + str(int(counter_SWAN)))
        plt.xlabel('number of iterations')
        plt.ylabel('Area under arch pop')
        i += 1
    plt.legend()
    plt.show()
    """