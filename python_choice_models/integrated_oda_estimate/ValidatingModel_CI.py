# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.



from scipy.stats import chi2
import math
import numpy as np
from scipy import optimize
from copy import deepcopy

def Chi_square_quantile(alpla, n):
    # alpha: confidence level 95%
    return chi2.ppf(alpla, df= n)

def upper_CI(hat_p, N_sample, N_prod, alpha):
    chi = Chi_square_quantile(alpha, N_prod)
    return hat_p + math.sqrt(chi * hat_p * (1 - hat_p)/N_sample)

def lower_CI(hat_p, N_sample, N_prod, alpha):
    chi = Chi_square_quantile(alpha, N_prod)
    return hat_p - math.sqrt(chi * hat_p * (1 - hat_p) / N_sample)


def SolveforValidatingModel(empirical_probabilities, num_transactions, confidence_level):
    N_prod = len(empirical_probabilities)

    bounds = []
    for prob in empirical_probabilities:
        L_CI = max(min(1,lower_CI(prob, num_transactions, N_prod, confidence_level)), 0)
        U_CI = min(max(0,upper_CI(prob, num_transactions, N_prod, confidence_level)), 1)
        bounds.append(tuple([L_CI, U_CI]))

    z = np.random.random(N_prod)
    res = optimize.linprog(z, A_eq=[[1 for i in range(N_prod)]], b_eq= [1], bounds=bounds)
    prob = res.x
    for i in range(len(prob)):
        if prob[i] > 1:
            prob[i] = 1
        elif prob[i] < 0:
            prob[i] = 0

    if sum(prob) != 1:
        sum_prob = sum(prob)
        for i in range(len(prob)):
            prob[i] = prob[i]/sum_prob
    return prob

## generate validating models based on the confidence interval
def ValidatingModel_CI(empirical_dict, confidence_level):
    #if dictionary contains list, then must use deep copy
    Validating_dict = deepcopy(empirical_dict)
    for offered_set, value in Validating_dict.items():
        for product in offered_set:
            if product in Validating_dict[offered_set]:
                continue
            else:
                Validating_dict[offered_set].update({product: 0.0})

    #check deviation
    sum_deivation = 0.0
    for offered_set in Validating_dict:
        if sum(offered_set) > 0:
            empirical_prob = np.zeros(len(offered_set))
            for key, val in Validating_dict[offered_set].items():
                if key != 'count_assor':
                    empirical_prob[offered_set.index(key)] = val
            num_transactions = Validating_dict[offered_set]['count_assor']

            validating_probs = SolveforValidatingModel(empirical_prob, num_transactions, confidence_level)
            for product in offered_set:
                Validating_dict[offered_set].update({product: validating_probs[offered_set.index(product)]})

            for prod in offered_set:
                # print(Validating_dict[offered_set][prod], empirical_dict[offered_set][
                # prod])
                if prod in empirical_dict[offered_set]:
                    sum_deivation += (Validating_dict[offered_set][prod] - empirical_dict[offered_set][
                    prod]) ** 2
                else:
                    sum_deivation += Validating_dict[offered_set][prod] ** 2
            sum_deivation = sum_deivation / len(offered_set)

    return Validating_dict, math.sqrt(sum_deivation)

