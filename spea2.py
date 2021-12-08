import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import subprocess
from numpy.linalg import norm as euclid_norm
from constraints import check_constraints
from SWAN_modeling import hs_modeling, surrogate_modeling, modeling
from shapely import affinity
from shapely.geometry import LineString
from visualization import example_create


class SPEA2_optimizer:

    def __init__(self,
                 pop_size,
                 arch_size,
                 max_iter,
                 mutat_rate_pop,
                 mutat_rate_individ,
                 domain):

        self.pop_size = pop_size
        self.arch_size = arch_size
        self.max_iter = max_iter
        self.domain = domain
        self.mutat_rate_pop = mutat_rate_pop
        self.mutat_rate_individ = mutat_rate_individ
        self.functions_to_optimize = []
        self.population = []  # list with shape (pop_size, desicion_space)
        self.archive = []
        self.hs_pop = []  # wave height in target point for pop
        self.hs_arch = []  # wave height in target point for arch
        self.Z_arch = []  # wh in every point
        self.Z_pop = []
        self.max_segm = 4
        self.min_segm = 1
        self.min_bw = 1
        self.max_bw = 3

    def objectives_calc(self, idx, type_of_pop):
        number_of_objectives = len(self.functions_to_optimize)
        objectives_val = []
        f1 = self.functions_to_optimize[0]

        if type_of_pop == 'arch':
            objectives_val.append(f1(self.archive[idx]))
            objectives_val.append(self.hs_arch[idx])

        elif type_of_pop == 'union':
            pop = self.population + self.archive
            hs = self.hs_pop + self.hs_arch
            objectives_val.append(f1(pop[idx]))
            objectives_val.append(hs[idx])

        return objectives_val

    # func for recognizing domaninance ind1 on ind2
    def dominance(self, k, l, type_of_pop):
        objective1 = self.objectives_calc(k, type_of_pop)
        objective2 = self.objectives_calc(l, type_of_pop)

        number_of_objectives = len(self.functions_to_optimize)

        is_equal = []
        is_dominating = []

        for i in range(number_of_objectives):
            if objective1[i] < objective2[i]:
                is_dominating.append(True)
            elif objective1[i] == objective2[i]:
                is_equal.append(True)
            else:
                is_dominating.append(False)

        return all(is_dominating) == True and len(is_dominating) > 0

    def strength(self, population, type_of_pop):
        size_of_pop = len(population)
        strength_pop = []  # list with form (pop_size, number of dominated individs)

        for i in range(size_of_pop):
            count = 0
            for j in range(size_of_pop):
                if j == i:
                    continue
                is_dominate = self.dominance(i, j, type_of_pop)
                if is_dominate:
                    count += 1
            strength_pop.append(count)

        return strength_pop

    def raw(self, population, type_of_pop):
        str_pop = self.strength(population, type_of_pop)
        size_of_pop = len(population)
        raw_pop = []

        for i in range(size_of_pop):
            count = 0
            for j in range(size_of_pop):
                if j == i:
                    continue
                is_dominate = self.dominance(j, i, type_of_pop)
                if is_dominate:
                    count += str_pop[j]
            raw_pop.append(count)

        return np.array(raw_pop)

    def density(self, population, type_of_pop):
        size_of_pop = len(population)
        dens_pop = []
        k = 1

        if type_of_pop == 'arch':
            f1 = self.functions_to_optimize[0]
            f2 = self.hs_arch
        elif type_of_pop == 'union':
            f1 = self.functions_to_optimize[0]
            f2 = self.hs_pop + self.hs_arch

        for i in range(size_of_pop):
            dist = []
            first_point = np.array([f1(population[i]), f2[i]])
            for j in range(size_of_pop):
                if j == i:
                    continue
                second_point = np.array([f1(population[j]), f2[j]])
                dist.append(euclid_norm(first_point - second_point))
            sorted_dist = np.sort(dist)
            dens_pop.append(1 / (sorted_dist[k] + 2))

        return dens_pop

    def fitness(self, population, type_of_pop):
        R = self.raw(population, type_of_pop)
        D = self.density(population, type_of_pop)
        F = R + D

        return list(F)

    def remove_dublicates(self, obj):
        idx_no_dublicates = []
        new_obj = []

        for i, element in enumerate(obj):
            if element not in new_obj:
                idx_no_dublicates.append(i)
            new_obj.append(element)

        return idx_no_dublicates

    def environmental_selection(self, union_pop, union_fit, union_hs):
        idx_no_dubl = self.remove_dublicates(union_hs)
        union_pop = [union_pop[idx] for idx in idx_no_dubl]
        union_fit = [union_fit[idx] for idx in idx_no_dubl]
        union_hs = [union_hs[idx] for idx in idx_no_dubl]

        f1 = self.functions_to_optimize[0]
        arch_idx_selected = [i for i, individ_fit in enumerate(union_fit) if individ_fit < 1.0]

        new_arch_pop = [union_pop[idx] for idx in arch_idx_selected]
        new_arch_hs = [union_hs[idx] for idx in arch_idx_selected]
        new_arch_len = len(new_arch_pop)
        difference = int(abs(new_arch_len - self.arch_size))

        if new_arch_len > self.arch_size:
            k = 1

            idx_for_delete = []
            distance_and_idx = []
            for i in range(new_arch_len):
                first_point = np.array(f1([new_arch_pop[i]]), new_arch_hs[i])
                dist = []
                for j in range(new_arch_len):
                    if j == i:
                        continue
                    second_point = np.array(f1([new_arch_pop[j]]), new_arch_hs[j])
                    dist.append(euclid_norm(first_point - second_point))
                sort_dist = sorted(list(dist))
                distance_and_idx.append((sort_dist[k], i))
            sorted_distance_and_idx = sorted(distance_and_idx)

            for i in range(difference):
                idx_for_delete.append(sorted_distance_and_idx[i][1])

            arch_idx_selected = [idx for idx in arch_idx_selected if idx not in idx_for_delete]

        elif new_arch_len < self.arch_size:
            fit_with_idx = [(individ_fit, i) for i, individ_fit in enumerate(union_fit)]
            union_sorted = sorted(fit_with_idx)

            idx_for_add = []
            count = 0
            for individ in union_sorted:
                if individ[0] >= 1.0:
                    count += 1
                    idx_for_add.append(individ[1])
                if count == difference:
                    break

            arch_idx_selected = arch_idx_selected + idx_for_add

        new_arch = [union_pop[idx] for idx in arch_idx_selected]
        new_arch_hs = [union_hs[idx] for idx in arch_idx_selected]

        return new_arch, new_arch_hs

    def selection(self, pop, fit):
        selected = []
        sum_fit = sum(fit)
        probab = (np.flip(np.array(fit) / sum_fit)).tolist()
        sort_pop = sorted(pop)

        for i in range(self.pop_size):
            selected.append(random.choices(sort_pop, weights=probab)[0])
            ind = sort_pop.index(selected[i])
            del probab[ind], sort_pop[ind]

        return selected

    def point_crossover(self, ind1, ind2, losted):
        out1, out2 = [], []
        losted1, losted2 = losted[0], losted[1]

        max_point = min(len(ind1), len(ind2))
        points = random.sample(range(1, max_point), max_point - 1)
        if max(len(ind1), len(ind2)) > max_point:
            points = random.sample(range(1, max_point + 1), max_point)

        for point in points:
            ind_new1 = [ind1[:point] + ind2[point:]]
            ind_new2 = [ind2[:point] + ind1[point:]]

            new_lost1 = ind_new1 + losted1
            new_lost2 = ind_new2 + losted2

            if not(check_constraints(new_lost1) or check_constraints(new_lost2)):
                out1 = new_lost1
                out2 = new_lost2
                break

        return out1, out2

    def lower_crossover(self, pair):
        individ1 = pair[0]
        individ2 = pair[1]
        out1, out2 = [], []

        configure1 = random.sample(individ1, len(individ1))
        configure2 = random.sample(individ2, len(individ2))

        for ind1, ind2 in zip(configure1, configure2):
            losted1 = [ind for ind in individ1 if ind != ind1]
            losted2 = [ind for ind in individ2 if ind != ind2]

            out1, out2 = self.point_crossover(ind1, ind2, [losted1, losted2])

            if not(check_constraints(out1) or check_constraints(out2)):
                break

        if check_constraints(out1) or check_constraints(out2):
            return False

        else:
            return [out1, out2]

    """
    def middle_crossover(self, pair):
        individ1 = pair[0]
        individ2 = pair[1]
        out1, out2 = [], []

        ind_more_one_segments1 = [ind for ind in individ1 if int((len(ind) - 2) / 2) > 1]
        ind_more_one_segments2 = [ind for ind in individ2 if int((len(ind) - 2) / 2) > 1]

        if len(ind_more_one_segments1) == 0 or len(ind_more_one_segments2) == 0 :
            out1, out2 = self.lower_crossover(pair)

        else:
            losted1 = [ind for ind in individ1 if ind not in ind_more_one_segments1]
            losted2 = [ind for ind in individ2 if ind not in ind_more_one_segments2]
            for ind1 in ind_more_one_segments1:
                for ind2 in ind_more_one_segments2:



        if constraints(out1) or constraints(out2):
            out1, out2 = self.lower_crossover(pair)

        return [out1, out2]
    """

    def upper_crossover(self, pair):
        individ1 = pair[0]
        individ2 = pair[1]
        out1, out2 = [], []

        if max(len(individ1), len(individ2)) == 1:
            out = self.lower_crossover(pair)

        elif max(len(individ1), len(individ2)) != 1:
            max_point = min(len(individ1), len(individ2))
            points = random.sample(range(1, max_point + 1), max_point)
            for point in points:
                out1 = individ1[:point] + individ2[point:]
                out2 = individ2[:point] + individ1[point:]
                out = [out1, out2]
                if not (check_constraints(out1) or check_constraints(out2)):
                    break

        if check_constraints(out1) or check_constraints(out2):
            out = self.lower_crossover(pair)

        return out

    def crossover(self, pair):
        out = self.upper_crossover(pair)

        return out

    def center_mass_displacement(self, individ):
        bw_for_mutat = random.randint(1, len(individ))
        ind_for_mutat = random.sample(individ, bw_for_mutat)
        new_individ = [ind for ind in individ if ind not in ind_for_mutat]

        for ind in ind_for_mutat:
            new_ind = []
            eps = 200
            delta_x = random.randint(-eps, eps)
            delta_y = random.randint(-eps, eps)
            for i, gen in enumerate(ind):
                if i % 2 == 0:
                    new_ind.append(ind[i] + delta_x)
                elif (i + 1) % 2 == 0:
                    new_ind.append(ind[i] + delta_y)
            new_individ.append(new_ind)

        return new_individ

    def add_delete_bw(self, individ):
        new_individ = individ
        num_of_bw = len(individ)
        r = random.random()

        if r < 0.5 and num_of_bw != self.max_bw:
            case_add = {
                '1': [1, 2],
                '2': [1]
            }
            num_new_bw = random.choice(case_add[str(num_of_bw)])
            new_bw = [self.create_breakwater(random.randint(self.min_segm, self.max_segm)) for _ in range(num_new_bw)]
            new_individ = individ + new_bw

        elif r > 0.5 and num_of_bw != self.min_bw:
            case_delete = {
                '2': [1],
                '3': [1, 2]
            }
            num_delete_bw = random.choice(case_delete[str(num_of_bw)])
            new_individ = random.sample(individ, num_delete_bw)

        return new_individ

    def whole_turn(self, individ):
        bw_for_mutat = random.randint(1, len(individ))
        ind_for_mutat = random.sample(individ, bw_for_mutat)
        new_individ = [ind for ind in individ if ind not in ind_for_mutat]

        for ind in ind_for_mutat:
            new_ind = []
            ind_for_rotate = LineString([(x, y) for x, y in zip(ind[::2], ind[1:][::2])])
            angle = random.randint(0, 360)
            new_individ_line = affinity.rotate(ind_for_rotate, angle, 'center')
            for x, y in zip(list(new_individ_line.xy[0]), list(new_individ_line.xy[1])):
                new_ind += [int(x)]
                new_ind += [int(y)]
            new_individ.append(new_ind)

        return new_individ

    def upper_mutation(self, individ):
        upper_num_mutation = 3  # количество операторов мутации на верхнем уровне
        weights = [0.4, 0.4, 0.2]
        mutat_case = random.choices(range(1, upper_num_mutation + 1), weights)[0]

        if mutat_case == 1:
            new_individ = self.center_mass_displacement(individ)
            while check_constraints(new_individ):
                new_individ = self.center_mass_displacement(individ)
        elif mutat_case == 2:
            new_individ = self.whole_turn(individ)
            while check_constraints(new_individ):
                new_individ = self.whole_turn(individ)
        elif mutat_case == 3:
            new_individ = self.add_delete_bw(individ)
            while check_constraints(new_individ):
                new_individ = self.add_delete_bw(individ)

        return new_individ

    def point_displacement(self, individ):
        bw_for_mutat = random.randint(1, len(individ))
        ind_for_mutat = random.sample(individ, bw_for_mutat)
        new_individ = [ind for ind in individ if ind not in ind_for_mutat]

        for ind in ind_for_mutat:
            new_ind = []
            eps = 220
            displacement = random.randint(-eps, eps)
            for i, gen in enumerate(ind):
                r = random.random()
                if r < 0.5:
                    new_ind.append(ind[i] + displacement)
                else:
                    new_ind.append(ind[i])
            new_individ.append(new_ind)

        return new_individ

    def add_delete_segment(self, individ):
        new_individ = []
        area_local = [self.domain[0], self.domain[1]]
        add_ratio = 0.5
        delete_ratio = 0.5

        for ind in individ:
            l = len(ind)
            number_of_segments = int((l - 2) / 2)

            add_or_delete = random.random()

            if add_or_delete < add_ratio and number_of_segments != 4:
                case_add = {
                    '1': [1, 2, 3],
                    '2': [1, 2],
                    '3': [1]
                }
                num_new_segments = random.choice(case_add[str(number_of_segments)])
                num_new_points = int(2 * num_new_segments)
                area = area_local * int(num_new_points / 2)
                new_ind = ind + [np.random.randint(low=interval[0], high=interval[1] + 1) for interval in area]
                new_individ.append(new_ind)

            elif add_or_delete > delete_ratio and number_of_segments != 1:
                case = {
                    '6': [1],
                    '8': [1, 2],
                    '10': [1, 2, 3]
                }
                case_idx = {
                    '0': [0, 1],
                    '1': [2, 3],
                    '2': [4, 5],
                    '3': [6, 7],
                    '4': [8, 9]
                }

                size = random.choice(case[str(l)])
                num_for_delete = random.sample(range(0, int(l / 2)), size)
                idx_for_delete = []
                for num in num_for_delete:
                    idx_for_delete += case_idx[str(num)]

                left_individ = []
                for i, coord in enumerate(ind):
                    if i not in idx_for_delete:
                        left_individ.append(ind[i])

                new_individ.append(left_individ)

            else:
                new_individ.append(ind)

        return new_individ

    def lower_mutation(self, individ):
        lower_num_mutation = 2  # количество операторов мутации на верхнем уровне
        mutat_case = random.randint(1, lower_num_mutation)

        if mutat_case == 1:
            new_individ = self.point_displacement(individ)
            while check_constraints(new_individ):
                new_individ = self.point_displacement(individ)
        elif mutat_case == 2:
            new_individ = self.add_delete_segment(individ)
            while check_constraints(new_individ):
                new_individ = self.add_delete_segment(individ)

        return new_individ

    def mutation(self, individ):
        r = random.random()

        if r > 0.5:
            new_individ = self.upper_mutation(individ)
        else:
            new_individ = self.lower_mutation(individ)

        return new_individ

    def variation(self, union_pop, union_fit):
        union_fit_idx = [(fit, i) for i, fit in enumerate(union_fit)]
        sorted_fit = sorted(union_fit_idx)
        selected_pop = [union_pop[idx[1]] for idx in sorted_fit]

        new_pop = []
        mutat_pop = []

        for ind1 in selected_pop:
            for ind2 in selected_pop:
                if ind1 == ind2:
                    continue
                out = self.crossover([ind1, ind2])
                if out[0] and out[1]:
                    new_pop.append(out[0])
                    new_pop.append(out[1])
                    selected_pop.remove(ind1)
                    selected_pop.remove(ind2)
                    break

        mutat_part = int(self.mutat_rate_pop * len(new_pop))
        pop_for_mutat = random.sample(new_pop, mutat_part)
        [new_pop.remove(individ) for individ in pop_for_mutat]
        for individ in pop_for_mutat:
            mutat_pop.append(self.mutation(individ))

        out_pop = new_pop + mutat_pop

        return out_pop[:self.pop_size]

    def create_breakwater(self, bw_num):
        area_local = [self.domain[0], self.domain[1]]
        number_of_points = int(2 * bw_num) + 2
        area = area_local * int(number_of_points / 2)
        breakwater = [np.random.randint(low=interval[0], high=interval[1] + 1) for interval in area]

        return breakwater

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            number_of_bw = random.randint(self.min_bw, self.max_bw)
            individ = [self.create_breakwater(random.randint(self.min_segm, self.max_segm)) for _ in
                       range(number_of_bw)]
            while check_constraints(individ):
                individ = [self.create_breakwater(random.randint(self.min_segm, self.max_segm)) for _ in range(number_of_bw)]
            population.append(individ)

        return population


    def unique_add(self, individs):
        unique_individs = [ind for ind in individs if ind not in self.archive]
        _, hs_individs, counter_SWAN = hs_modeling(unique_individs)

        return unique_individs, hs_individs, counter_SWAN


    def optimize(self, init_pop, func_to_optimize, type_modeling):
        Z_for_exp = []
        pop_for_exp = []
        HS = []

        self.functions_to_optimize = func_to_optimize
        self.population = init_pop
        #self.archive = []
        self.archive = [[[1176, 539, 735, 828]], [[1176, 539, 735, 828], [768, 666, 1070, 321]], [[1176, 560, 850, 747], [768, 666, 1070, 321]], [[1155, 742, 953, 691]], [[1073, 674, 850, 747]], [[1176, 560, 850, 747]], [[953, 742, 1156, 709]], [[1176, 539, 626, 872]], [[1053, 842, 879, 949]], [[1073, 674, 855, 760]], [[845, 742, 1156, 709]], [[1060, 750, 844, 782]], [[1420, 827, 1284, 680]], [[909, 721, 735, 828]], [[953, 742, 1156, 692]], [[1073, 674, 735, 828]], [[845, 742, 1156, 709], [768, 666, 1070, 321]], [[1176, 560, 850, 747, 1198, 644], [768, 666, 1070, 321]], [[968, 704, 626, 872, 1221, 644], [892, 404, 585, 744]], [[968, 704, 626, 872, 1221, 644], [864, 382, 613, 766]]]
        _, self.hs_pop, counter_SWAN = hs_modeling(self.population)
        _, self.hs_arch, counter_SWAN = hs_modeling(self.archive)

        #Z_for_exp += Z
        #pop_for_exp += self.population
        counter_SWAN = 653

        arch_counter = []
        #arch_counter.append((self.population, counter_SWAN))

        number = 10
        border = counter_SWAN + self.pop_size
        it = 0

        arch_history = []
        while len(arch_counter) != number:
            print('Swan calc = ' + str(counter_SWAN))
            print('ITER = ' + str(it))

            union_pop = self.population + self.archive
            union_fit = self.fitness(union_pop, 'union')
            union_hs = self.hs_pop + self.hs_arch

            individs_to_add, _ = self.environmental_selection(union_pop, union_fit, union_hs)
            unique_individs, hs_individs, counter = self.unique_add(individs_to_add)



            arch_history.append(self.archive)

            if counter_SWAN >= border:
                arch_counter.append((self.archive, counter_SWAN))
                border += self.pop_size

            mating_pool = union_pop
            self.population = self.variation(mating_pool, union_fit)
            _, self.hs_pop, counter = type_modeling(self.population)

            #Z_for_exp += Z
            #pop_for_exp += self.population

            counter_SWAN += counter
            it += 1

        #_ = [example_create(Z_for_exp[i], pop_for_exp[i], i, label=True) for i, __ in enumerate(Z_for_exp)]
        #_ = [example_create(Z_for_exp[i], pop_for_exp[i], i, label=False) for i, __ in enumerate(Z_for_exp)]

        #HS = [Z_for_exp[i][23,40] for i in range(len(Z_for_exp))]

        return self.archive, arch_counter, it, self.population, HS