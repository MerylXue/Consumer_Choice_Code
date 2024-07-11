# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import json
import time

import numpy as np

from python_choice_models.transactions.base import Transaction
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator


from python_choice_models.integrated_oda_estimate.ValidatingModel import GenerateOutofSampleTransactions, SequenceValidatingModels, SampleTrainingTransactions
from python_choice_models.integrated_oda_estimate.SmoothingParameter_Multiple import get_alpha_multiple
from python_choice_models.integrated_oda_estimate.Bootstrap import run_with_os, Validate_Model_str
from python_choice_models.optimization.regret_optimization import Min_Max_Regret_OS
from python_choice_models.integrated_oda_estimate.ODA_optimal_alpha import Validate_ODA_alpha_dict

def Optimal_Operational_Statistics(file_name, beta, K_train, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    N_prod = len(products) - 1
    K_sample = 10
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    empirical_dict = EmpiricalEstimator(in_sample_transactions).empirical_dict
    Sequenced_validating_models, Sorted_deviation, Sorted_likelihood = SequenceValidatingModels(beta, N_prod,
                                                                                              K_sample - 1,
                                                                                                empirical_dict,
                                                                                                in_sample_transactions)
    K_sample_r = len(Sequenced_validating_models)


    K_mrmse = [[] for s in range(K_sample_r)]
    K_mae = [[] for s in range(K_sample_r)]
    K_model_name = [[] for s in range(K_sample_r)]
    K_method_name = [[] for s in range(K_sample_r)]
    for s in range(K_sample_r):
        for k in range(K_train):
            # results = []
            model_name = []
            method_name = []
            mrmses = []
            maes = []
            validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                       N_prod,
                                                                                       Sequenced_validating_models[s])

            for estimation_method, method_info in list(estimators.items()):
                for model, model_info in list(method_info['models'].items()):
                    # print('\tEST.\tMODEL\tH-RMSE\tH-AE\tS-RMSE.\tS-AE\tTIME')

                    # print(estimation_method, model, alpha)
                    # result = pool.apply_async(show_row_oda,args=(estimation_method, model, validating_transactions,
                    #                                             validating_prob, products))
                    # record the result
                    rmse_ground, ae_ground = run_with_os(estimation_method, model, validating_transactions,
                                                         validating_prob, products, estimators)
                    method_name.append(estimation_method)
                    model_name.append(model)
                    # results.append(result)
                    mrmses.append(rmse_ground)
                    maes.append(ae_ground)

            # K_results.append(results)
            K_mrmse[s].append(mrmses)
            K_mae[s].append(maes)
            K_model_name[s].append(model_name)
            K_method_name[s].append(method_name)

    best_model = [dict() for s in range(K_sample)]
    # find the best os for each sampled validating model
    for s in range(K_sample_r):
        for k in range(K_train):
            mrmse_lst = K_mrmse[s][k]
            mae_lst = K_mae[s][k]
            method_name = K_method_name[s][k]
            model_name = K_model_name[s][k]
            for idx in range(len(mae_lst)):
                s_mrmse = mrmse_lst[idx]
                s_mae = mae_lst[idx]
                method = method_name[idx]
                model = model_name[idx]

                if model in best_model[s]:
                    if method in best_model[s][model]:
                        val1 = best_model[s][model][method]['mrmse']
                        val1 += s_mrmse
                        best_model[s][model][method].update({'mrmse': val1})

                        val2 = best_model[s][model][method]['mae']
                        val2 += s_mae
                        best_model[s][model][method].update({'mae': val2})
                    else:
                        best_model[s][model].update({method: {'mrmse': s_mrmse, 'mae': s_mae}})
                else:
                    best_model[s].update({model: {method: {'mrmse': s_mrmse, 'mae': s_mae}}})


    optimal_model = ['' for s in range(K_sample_r)]
    optimal_method = ['' for s in range(K_sample_r)]
    for s in range(K_sample_r):
        min_mae = 100
        for key_model in best_model[s]:
            for key_method in best_model[s][key_model]:
                val1 = best_model[s][key_model][key_method]['mae'] / K_train
                if val1 < min_mae:
                    optimal_model[s] = key_model
                    optimal_method[s] = key_method
                    min_mae = val1

    mrmse_table = np.zeros((K_sample, K_sample))
    mae_table = np.zeros((K_sample, K_sample))


    for k in range(K_sample_r):  # validating model (as row)
        for s in range(K_sample_r):  # optimal os (as col)
            val1 = best_model[k][optimal_model[s]][optimal_method[s]]['mrmse'] / K_train
            val2 = best_model[k][optimal_model[s]][optimal_method[s]]['mae'] / K_train
            mrmse_table[k][s] = val1
            mae_table[k][s] = val2

    return optimal_model, optimal_method, mrmse_table, mae_table, Sequenced_validating_models, \
           Sorted_deviation, Sorted_likelihood

def Max_Regret_OS_alpha_dict(file_name, beta, K_train, estimators):
    # K_sample = 10
    print("start max regret os alpha dict")
    K_sample = 10
    optimal_model, optimal_method, rmse_table, ae_table, Sequenced_validating_models, \
    Sorted_deviation, Sorted_likelihood = Optimal_Operational_Statistics(file_name,  beta,  K_train, estimators)
    regret_error = np.zeros((K_sample, K_sample))

    for k in range(K_sample):
        min_regret = min(rmse_table[k]) #min_error(model, os) for all os
        for s in range(K_sample):
            regret_error[k][s] = rmse_table[k][s] - min_regret

    #calcalte the maximum regret
    max_regret_error = np.zeros((K_sample, K_sample)) # for each os, find the max

    for s in range(K_sample):
        for k in range(K_sample):
            sub_table = regret_error[0:k + 1,s]
            max_regret_error[k][s] = np.max(sub_table)

    return Sorted_deviation, Sequenced_validating_models, max_regret_error, optimal_model, optimal_method


def optimal_alpha(input_file, Sequenced_validating_models, optimal_model, optimal_method):
    input_file = open(input_file, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    # N_prod = len(products) - 1
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])

    alpha_dict = get_alpha_multiple(optimal_method, optimal_model, products, in_sample_transactions, Sequenced_validating_models)

    return alpha_dict



def Validate_MinMaxRegret_Model_alphadict(input_file, beta,  K_train_os, estimators):

    # Optimal_Operational_Statistics(input_file, beta, K_sample, K_train_os, estimators)
    t1 = time.time()
    Sorted_deviation, Sequenced_validating_models, max_regret_error, optimal_model, optimal_method = Max_Regret_OS_alpha_dict(input_file, beta,K_train_os, estimators)
    #
    # optimal os chosen by min max regret
    K_sample = 10
    optimal_os_regret = Min_Max_Regret_OS(K_sample, max_regret_error)
    outline = ['%s' % optimal_model[optimal_os_regret]]
    # method_file.write(','.join(outline) + '\n')
    # print(ae_table)
    # print(regret_error)
    optimal_alpha_dict = optimal_alpha(input_file, Sequenced_validating_models, optimal_model[optimal_os_regret], optimal_method[optimal_os_regret])

    error_dict = Validate_ODA_alpha_dict(input_file, optimal_model[optimal_os_regret],
                                                                                 optimal_method[optimal_os_regret],
                                                                                 optimal_alpha_dict, estimators)
    t2 = time.time()
    error_dict.update({"time": t2 - t1})
    error_dict.update({"num_iter": 0})
    return error_dict


## the oda method without interpolating with empirical estimation
def Validate_MinMaxRegret_Model_str(input_file, beta, K_train_os, estimators):
    t1 = time.time()
    Sorted_deviation, Sequenced_validating_models, max_regret_error, optimal_model, optimal_method = Max_Regret_OS_alpha_dict(input_file, beta, K_train_os, estimators)
    #
    # optimal os chosen by min max regret
    K_sample = 10
    optimal_os_regret = Min_Max_Regret_OS(K_sample, max_regret_error)
    error_dict = Validate_Model_str(input_file, optimal_model[optimal_os_regret], optimal_method[optimal_os_regret], estimators)
    t2 = time.time()
    error_dict.update({"time": t2 - t1})
    error_dict.update({"num_iter": 0})
    return error_dict