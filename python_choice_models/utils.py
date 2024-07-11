# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

from numpy import array
from math import log
from itertools import chain, combinations, permutations
from numba import jit
import numpy
import json
import math
import itertools
ZERO_LOWER_BOUND = 1e-6
ONE_UPPER_BOUND = 1.0 - ZERO_LOWER_BOUND

FINITE_DIFFERENCE_DELTA = 1e-7


@jit
def safe_log_array(old_array):
    new_array = []
    for number in old_array:
        new_array.append(safe_log(number))
    return new_array


@jit
def safe_log(x):
    # This is to deal with infeasible optimization methods (those who don't care about evaluating objective function
    # inside constraints, this could cause evaluating outside log domain)
    if x > ZERO_LOWER_BOUND:
        return log(x)
    log_lower_bound = log(ZERO_LOWER_BOUND)
    a = 1 / (3 * ZERO_LOWER_BOUND * (3 * log_lower_bound * ZERO_LOWER_BOUND)**2)
    b = ZERO_LOWER_BOUND * (1 - 3 * log_lower_bound)
    return a * (x - b) ** 3


def finite_difference(function):
    def derivative(x):
        h = FINITE_DIFFERENCE_DELTA
        gradient = []
        x = list(x)
        for i, parameter in enumerate(x):
            plus = function(x[:i] + [parameter + h] + x[i + 1:])
            minus = function(x[:i] + [parameter - h] + x[i + 1:])
            gradient.append((plus - minus) / (2 * h))
        return array(gradient)
    return derivative


def generate_n_random_numbers_that_sum_one(n):
    distribution = [numpy.random.uniform(0, 1) for _ in range(n)]
    total = sum(distribution)

    for i in range(len(distribution)):
        distribution[i] = distribution[i] / total

    return distribution


def generate_n_equal_numbers_that_sum_one(n):
    head = [1.0 / n for _ in range(n - 1)]
    return head + [1.0 - sum(head)]


def generate_n_equal_numbers_that_sum_m(n, m):
    return [x * m for x in generate_n_equal_numbers_that_sum_one(n)]


def generate_n_random_numbers_that_sum_m(n, m):
    return [x * m for x in generate_n_random_numbers_that_sum_one(n)]


def time_for_optimization(partial_time, total_time, profiler):
    if not partial_time:
        return max(total_time - profiler.duration(), 0.01)
    return max(min(partial_time, total_time - profiler.duration()), 0.01)


def rindex(a_list, a_value):
    return len(a_list) - a_list[::-1].index(a_value) - 1


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
###############################32022-08###########################################3

def powerset_no_empty(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def powerset_seq(iterable):
    s = list(iterable)
    if not 0 in s:
        return list(chain.from_iterable(permutations(s, r) for r in range(1,len(s) + 1)))
    else:
        s2 = [i for i in s if not i == 0]
        comb = list(chain.from_iterable(permutations(s2, r) for r in range(1,len(s2) + 1)))
        return comb + [tuple([0])] +  [tuple(list(lst) + [0]) for lst in comb]

    # return chain.from_iterable(permutations(s, r) for r in range(1,len(s) + 1)) ## no empty


def powerset_seq_with_empty(iterable):
    s = list(iterable)
    if not 0 in s:
        # print(s, list(chain.from_iterable(permutations(s, r) for r in range(len(s) + 1))))
        return list(chain.from_iterable(permutations(s, r) for r in range(len(s) + 1)))
    else:
        s2 = [i for i in s if not i == 0]
        comb = list(chain.from_iterable(permutations(s2, r) for r in range(len(s2) + 1)))
        return comb + [tuple([0])] + [tuple(list(lst) + [0]) for lst in comb]

#generate all possible permutation in iterable with length level
def powersest_seq_level_k(iterable, k):
    s = list(iterable)
    if not 0 in s:
        return list(permutations(s, k))
    elif k == len(s):
        s2 = [i for i in s if not i == 0]
        comb = list(permutations(s2, len(s2)))
        return [tuple(list(lst) + [0]) for lst in comb]
    else:
        s2 = [i for i in s if not i == 0]

        comb = list(permutations(s2, k))
        # print(comb)
        com2 = list(permutations(s2, k-1))
        # print(comb, com2)
        return comb + [tuple(list(lst) + [0]) for lst in com2]

    # return chain.from_iterable(permutations(s, r) for r in range(len(s) + 1))
###############################32022-08###########################################3
class User(object):

    def __init__(self, name):
        self.name = name

def addtwodimdict(thedict,  key_b, val):
    thedict.update({key_b: val})


def addtwodimdict2(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

class UserEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, User):
            return obj.name
        return json.JSONEncoder.default(self, obj)

def update_dict_lst(key, error, dict0):
    if key in dict0:
        val_lst = dict0[key]
        val_lst.append(error)

        dict0.update({key: val_lst})
    else:
        dict0.update({key: [error]})
    return dict0



def add_details_to_dict(model, method, error_dict, detail_dict):
    if model in detail_dict:
        if method in detail_dict[model]:
            idx = max(list(detail_dict[model][method].keys()))
            detail_dict[model][method].update({idx+1 :{'rmse_in': error_dict['rmse_in'], 'ae_in': error_dict['ae_in'],
                                           'mrmse_in': error_dict['mrmse_in'], 'mae_in': error_dict['mae_in'],
                                           'rmse_out': error_dict['rmse_out'], 'ae_out': error_dict['ae_out'],
                                           'mrmse_out': error_dict['mrmse_out'], 'mae_out': error_dict['mae_out'],
                                           'rmse_all': error_dict['rmse_all'], 'ae_all': error_dict['ae_all'],
                                           'mrmse_all': error_dict['mrmse_all'], 'mae_all': error_dict['mae_all'],
                                           'AIC': error_dict['AIC'], 'BIC': error_dict['BIC'], 'chi2': error_dict['chi2'],
                                           'time': error_dict['time']}})
    else:
        detail_dict.update({model:
                                {method: {1: {'rmse_in': error_dict['rmse_in'], 'ae_in': error_dict['ae_in'],
                                                    'mrmse_in': error_dict['mrmse_in'], 'mae_in': error_dict['mae_in'],
                                                    'rmse_out': error_dict['rmse_out'], 'ae_out': error_dict['ae_out'],
                                                    'mrmse_out': error_dict['mrmse_out'],
                                                    'mae_out': error_dict['mae_out'],
                                                    'rmse_all': error_dict['rmse_all'], 'ae_all': error_dict['ae_all'],
                                                    'mrmse_all': error_dict['mrmse_all'],
                                                    'mae_all': error_dict['mae_all'],
                                                    'AIC': error_dict['AIC'], 'BIC': error_dict['BIC'],
                                                    'chi2': error_dict['chi2'],
                                                    'time': error_dict['time']}}}})

    return detail_dict
def update_error_by_dict_lst(model, method, error_dict, avg_dict):
    if model in avg_dict:
        if method in avg_dict[model]:
            for key in error_dict.keys():
                # print(avg_dict[model][method])
                avg_dict[model][method] = update_dict_lst(key, error_dict[key], avg_dict[model][method])
            if len(error_dict.keys()) < 12 + 3 + 1:
                print("Miss errors in error dict!")
        else:
            avg_dict[model].update(
                {method: {'rmse_in': [error_dict['rmse_in']], 'ae_in': [error_dict['ae_in']],
                          'mrmse_in': [error_dict['mrmse_in']], 'mae_in': [error_dict['mae_in']],
                          'rmse_out': [error_dict['rmse_out']], 'ae_out': [error_dict['ae_out']],
                          'mrmse_out': [error_dict['mrmse_out']], 'mae_out': [error_dict['mae_out']],
                          'rmse_all': [error_dict['rmse_all']], 'ae_all': [error_dict['ae_all']],
                          'mrmse_all': [error_dict['mrmse_all']], 'mae_all': [error_dict['mae_all']],
                          'AIC': [error_dict['AIC']], 'BIC': [error_dict['BIC']], 'chi2': [error_dict['chi2']],
                          'time': [error_dict['time']]}})
    else:
        avg_dict.update({model:
                             {method: {'rmse_in': [error_dict['rmse_in']], 'ae_in': [error_dict['ae_in']],
                                       'mrmse_in': [error_dict['mrmse_in']], 'mae_in': [error_dict['mae_in']],
                                       'rmse_out': [error_dict['rmse_out']], 'ae_out': [error_dict['ae_out']],
                                       'mrmse_out': [error_dict['mrmse_out']], 'mae_out': [error_dict['mae_out']],
                                       'rmse_all': [error_dict['rmse_all']], 'ae_all': [error_dict['ae_all']],
                                       'mrmse_all': [error_dict['mrmse_all']], 'mae_all': [error_dict['mae_all']],
                                       'AIC': [error_dict['AIC']], 'BIC': [error_dict['BIC']],
                                       'chi2': [error_dict['chi2']],
                                       'time': [error_dict['time']]}}})
    return avg_dict

def statistics_dict_lst_update(key, dict0, K_train, n_samples):
    # avg_dict = deepcopy(dict0)
    if key in dict0:
        val_lst = dict0[key]
        avg_lst = numpy.average(numpy.array(val_lst))
        var_lst = numpy.var(numpy.array(val_lst))
        sample_var = numpy.zeros(K_train)
        for t in range(K_train):
            sample_var[t] = numpy.var(numpy.array(val_lst[t*n_samples:(t+1)*n_samples]))

        avg_var_lst = numpy.average(sample_var)
        max_lst = max(val_lst)
        min_lst = min(val_lst)


        dict0.update({key: {'avg': avg_lst, 'all_var': var_lst, 'avg_var': avg_var_lst, 'max': max_lst, 'min': min_lst}})
    else:
        dict0.update({key: {'avg': -1, 'all_var': -1, 'avg_var': -1, 'max': -1, 'min': -1}})
    return dict0


def statistics_error_samples(avg_error_dict, K_train, n_samples):
    for model in avg_error_dict:
        for method in avg_error_dict[model]:
            for key in avg_error_dict[model][method].keys():
                avg_error_dict[model][method] = statistics_dict_lst_update(key,avg_error_dict[model][method], K_train, n_samples)
    return avg_error_dict


def assort_list_to_array(assortment, N_prod):
    a1 = numpy.zeros(N_prod + 1, int)
    a1[0] = 1
    for prod in assortment:
        a1[prod] = 1

    return a1

def array_to_assort_list(array, N_prod):
    return [i+1 for i in range(N_prod) if array[i] == 1]

#
# def Lattice(A, B):  # A<B
#     if set(B).intersection(set(A)) != set(A):
#         print("A does not belongs to B")
#
#     diff_set = list(set(B).difference(set(A)))
#     set_Assor = [A]
#     for i in range(1, len(diff_set) + 1):
#         permu_list = combinations(diff_set, i)
#         for k in permu_list:
#             # print(list(k))
#             set_Assor.append(A + list(k))
#
#     return set_Assor

def Lattice(A,B):
    if set(B).intersection(set(A)) != set(A):
        print("A does not belongs to B")

    diff_set = list(set(B).difference(set(A)))
    # set_Assor = [A]
    permu_list = chain.from_iterable(combinations(diff_set,r) for r in range(len(diff_set)+1))
    # for k in permu_list:
    #     set_Assor.append(A + list(k))
    # set_Assor = [A] + [*map(lambda x: A + list(x), permu_list)]
    return [*map(lambda x: A + list(x), permu_list)]

def CalculateAdvDistance(index_dict, probabilities, assort_list):
    n_products = len(probabilities[0]) - 1
    M = [i + 1 for i in range(n_products)]
    obj = 1e-6
    for B in index_dict:
        idx_B =  index_dict[tuple(B)]
        if idx_B in assort_list:
            B_lst = array_to_assort_list(B, n_products)
            # generate the subset of B
            A_lst = []

            for i in range(1, len(B_lst) + 1):
                permu_list = itertools.combinations(B_lst, i)
                for k in permu_list:
                    # print(list(k))
                    A_lst.append(list(k))


            for i in range(1, n_products + 1):
                if probabilities[idx_B][i] > 0:
                    obj += math.log( probabilities[idx_B][i])
                if B[i-1] - probabilities[idx_B][i] > 0:
                    obj += math.log(B[i - 1] - probabilities[idx_B][i])
            if probabilities[idx_B][0] > 0:
                obj += math.log(probabilities[idx_B][0])
            if probabilities[idx_B][0] < 1:
                obj += math.log(1-probabilities[idx_B][0])

            # obj += math.log(1-sum( model.probabilities[idx_B]))

            lattice = Lattice(B_lst,M)
            # print(A)
            # print(B_lst)
            # print(lattice)
            idx_lst = []
            for k in range(len(lattice)):
                # print(tuple([1 if i in lattice[k] else 0 for i in range(1, self.n_products + 1)]))
                idx_k = index_dict[
                    tuple([1 if i in lattice[k] else 0 for i in range(1, n_products + 1)])]
                idx_lst.append((idx_k, k))

            for item in [0] + B_lst:
                nabla = 0
                for a in idx_lst:
                    nabla += (-1) ** (len(lattice[a[1]]) - len(B_lst)) * probabilities[a[0]][item]
                if nabla > 0:
                    obj += math.log(nabla)
                # opt_m.addConstr(LinExpr([(-1) ** (len(lattice[a[1]]) - len(B_lst)) for a in idx_lst],
                #                         [x[a[0], item] for a in idx_lst]) >= 0.0)

    return obj


#################Settings
NORMAL_SETTINGS = {
        'linear_solver_partial_time_limit': None,
        'non_linear_solver_partial_time_limit': None,
        'solver_total_time_limit': 1800.0,
    }

RANKED_LIST_SETTINGS = {
    'linear_solver_partial_time_limit': 300,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}

LATENT_CLASS_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}

