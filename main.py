import numpy as np
import constraints
import pickle

from spea2 import SPEA2_optimizer
from models.get_models import Surrogate, RealModel
from models.surrogate_models import deep_conv_net, assistant_net
from objectives import cost, wave_h


def save_experimental_data(data):
    save_file = open("results/data.pkl", "wb")
    pickle.dump(data, save_file)
    save_file.close()


if __name__ == '__main__':
    PIX_TARGET = constraints.PIX_TARGET
    REV_POINT = constraints.REV_POINT

    TARGET = constraints.TARGET
    x = constraints.x
    y = constraints.y
    X, Y = np.meshgrid(x, y)
    Z = constraints.Z

    func_to_optimize = [cost, wave_h]
    spea2 = SPEA2_optimizer(pop_size=5,
                            arch_size=2,
                            max_iter=3,
                            mutat_rate_pop=0.35,
                            mutat_rate_individ=1.0,
                            domain=[[0, 2075],  # x-axis
                                    [0, 1450]],  # y-axis
                            )

    surr = Surrogate(surrogate=deep_conv_net(),
                     assistant=assistant_net(),
                     prepared_weight=['models/paper_weight/surrogate_1_5k_dataset',
                                      'models/paper_weight/assistant_1_5k_dataset', ])

    real = RealModel()

    history = spea2.optimize(func_to_optimize,
                             15,
                             surr,
                             real,
                             exploration_phase=True,
                             pretrained=True)

    save_experimental_data(history)
