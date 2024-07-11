# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from copy import deepcopy
import random
import numpy as np
import math

from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.ValidatingModel_CI import ValidatingModel_CI


def ValidatingModelLikelihood(Validating_dict, in_sample_transactions):
    L = 1
    for transaction in in_sample_transactions:
        offered_set = transaction.offered_products
        product = transaction.product

        prob = Validating_dict[tuple(offered_set)][product]
        L = L * prob

    # print(L)
    return L

## return a sequence of validating models and the correponding deviation
def SequenceValidatingModels(beta, n, K, empirical_dict, in_sample_transactions):
    #Sample K validating model
    ValidatingModel_lst = []
    #empirical model
    Validating_dict, deviation = ValidatingModel_CI(empirical_dict, 0)
    likelihood = ValidatingModelLikelihood(Validating_dict, in_sample_transactions)
    ValidatingModel_lst.append((Validating_dict, deviation,likelihood))

    #slightly deviated model
    for k in range(K):
        Validating_dict, deviation = ValidatingModel_CI(empirical_dict, beta)
        likelihood = ValidatingModelLikelihood(Validating_dict, in_sample_transactions)
        ValidatingModel_lst.append((Validating_dict,deviation,likelihood))


    #Sequence the validating model in the range of ascending deviation
    deviation_sorted = sorted(ValidatingModel_lst, key = lambda a: a[1])
    Sequenced_Model = [a[0] for a in deviation_sorted]
    Sorted_deviation = [a[1] for a in deviation_sorted]
    Sorted_likelihood = [a[2] for a in deviation_sorted]


    return Sequenced_Model, Sorted_deviation, Sorted_likelihood



def GenerateOutofSampleTransactions(in_sample_transactions, n, Validating_dict):
    #copy list

    out_of_sample_transactions = deepcopy(in_sample_transactions)
    out_of_sample_transactions_prob = []
    for transaction in out_of_sample_transactions:

        offered_set = transaction.offered_products
        choice_prob = np.zeros(n + 1)
        offered_product_prob = []
        for product in offered_set:
            choice_prob[product] = Validating_dict[tuple(offered_set)][product]
            offered_product_prob.append(Validating_dict[tuple(offered_set)][product])

        if min(choice_prob) < 0:
            for i in range(len(choice_prob)):
                if choice_prob[i] < 0:
                    choice_prob[i] = 0
        if max(choice_prob) > 1:
            for i in range(len(choice_prob)):
                if choice_prob[i] > 1:
                    choice_prob[i] = 1

        if sum(choice_prob) != 1:
            sum_prob = sum(choice_prob)

            for i in range(len(choice_prob)):
                choice_prob[i] = choice_prob[i]/sum_prob

        # print(choice_prob)

        choice = np.random.choice(n + 1, 1, p=choice_prob)
        transaction.product = choice[0]
        transaction_prob = Transaction_Extend(offered_product_prob, offered_set)
        # tmp_dict['prob'] = offered_product_prob
        # tmp_dict['Offered products'] = offered_set
        out_of_sample_transactions_prob.append(transaction_prob)

    return out_of_sample_transactions, out_of_sample_transactions_prob


def SampleTrainingTransactions(in_sample_prob):
    in_sample_transactions = []
    for transaction in in_sample_prob:
        product = np.random.choice(transaction.offered_products, 1, p= transaction.prob)[0]
        in_sample_transaction = Transaction(product, transaction.offered_products)
        in_sample_transactions.append(in_sample_transaction)

    return in_sample_transactions