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
from experiments import CNN_experiment, swan_experiment



if __name__ == '__main__':

    PIX_TARGET = constraints.PIX_TARGET
    REV_POINT = constraints.REV_POINT

    TARGET = constraints.TARGET
    x = constraints.x
    y = constraints.y
    X, Y = np.meshgrid(x, y)
    Z = constraints.Z

    func_to_optimize = [cost, wave_h]
    spea2 = SPEA2_optimizer(pop_size=40,
                            arch_size=20,
                            max_iter=3,
                            mutat_rate_pop=0.35,
                            mutat_rate_individ=1.0,
                            domain=[[570, 1480],  # x-axis
                                    [200, 1100]],  # y-axis
                            )

    #init_pop = spea2.initialize_population()
    init_pop = [[[845, 742, 1156, 709], [768, 666, 1070, 321]], [[1176, 560, 850, 747]], [[1176, 560, 850, 747]], [[1176, 539, 626, 872]], [[1176, 539, 735, 828]], [[1073, 674, 855, 760], [768, 666, 1070, 321]], [[1176, 539, 735, 828]], [[953, 742, 879, 949]], [[1053, 842, 1156, 709]], [[1176, 827, 1284, 680]], [[1420, 539, 735, 828]], [[1060, 674, 735, 828]], [[968, 704, 626, 872, 1221, 644]], [[968, 704, 626, 872, 1221, 644]], [[1176, 539, 735, 828], [588, 747, 890, 402]], [[1139, 703, 735, 828]], [[845, 742, 1156, 709]], [[1176, 560, 850, 747, 1198, 644]], [[1060, 750, 844, 782], [768, 666, 1070, 321]], [[968, 704, 626, 872, 1221, 644], [864, 382, 613, 766]], [[968, 704, 626, 872, 1221, 644], [864, 382, 613, 766]], [[953, 742, 1156, 709], [768, 666, 1070, 321]], [[969, 1021, 630, 897, 1203, 618], [588, 747, 890, 402]], [[1176, 560, 850, 747, 1198, 644]], [[1222, 887, 898, 709]], [[1444, 539, 855, 853]], [[1155, 742, 953, 691], [892, 209, 585, 744]], [[967, 800, 1214, 692]], [[968, 704, 626, 872, 1221, 644], [595, 742, 897, 397]], [[1348, 1025, 626, 872, 1221, 644, 1408, 628, 1177, 931], [588, 747, 890, 402]], [[1073, 674, 850, 747], [1134, 415, 703, 571]], [[953, 721, 746, 960]], [[1270, 489, 1267, 713]], [[1176, 560, 850, 747, 1198, 644], [756, 655, 1081, 331]], [[1176, 687, 794, 1020]], [[1073, 750, 956, 782]], [[1176, 761, 775, 868]], [[768, 666, 1070, 321], [1176, 609, 899, 987]], [[1078, 842, 855, 757], [588, 644, 890, 402]], [[1034, 947, 832, 907]]]
    types = [modeling, hs_modeling]

    integrals, solution, population = CNN_experiment(init_pop, types)

    Z_swan, _, _ = hs_modeling(solution[0])
    Z_cnn, _, _ = hs_modeling(solution[1])

    _ = [plot_map(X, Y, Z_swan[i], individ) for i, individ in enumerate(solution[0])]
    print('END')
    _ = [plot_map(X, Y, Z_cnn[i], individ) for i, individ in enumerate(solution[1])]


    np.savetxt('exp_results/HV/hv_after_CNN_25.txt', integrals[0])
    np.savetxt('exp_results/HV/hv_after_CNN_50.txt', integrals[1])
    np.savetxt('exp_results/HV/hv_after_CNN_75.txt', integrals[2])

    """
    def error_pred(hs_cnn, hs_swan, pred):
        error_pred = []
        error_pred_class = []
        for i in range(len(pred)):
            error = abs(hs_cnn[i] - hs_swan[i])
            error_pred.append((error, pred[i][0][0]))
            if pred[i][0][0] < 0.027755102040816326:
                error_pred_class.append(error)

        return error_pred, np.mean(error_pred_class)


    pop = spea2.initialize_population()
    hs_cnn, maps, pred = surrogate_modeling(pop)
    hs_swan = hs_modeling(pop)[1]

    err_pr, error_clss = error_pred(hs_cnn, hs_swan, pred)
    for g in err_pr:
        print(g)

    print('error with classifier:' + str(error_clss))

    def mape(true, pred):
        T, P = [], []
        for i in range(len(true)):
            if true[i] != pred[i]:
                T.append(true[i])
                P.append(pred[i])
        true = np.array(T)
        pred = np.array(P)

        return np.mean(np.abs((true - pred) / true)) * 100


    matplotlib.use('tkagg')

    for i, map in enumerate(maps):
        plt.imshow(map.reshape(224,224))
        plt.title(err_pr[i])
        plt.show()

    plt.scatter(hs_swan, hs_cnn)
    plt.plot(hs_swan, hs_swan)
    plt.show()
    print(mape(hs_swan, hs_cnn))
    """



