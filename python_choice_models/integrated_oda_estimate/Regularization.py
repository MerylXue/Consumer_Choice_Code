# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import json
import os
import sys
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/src/')

from python_choice_models.settings import Settings
from python_choice_models.integrated_oda_estimate.Bootstrap import GenerateOutofSampleTransactions
from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator
from python_choice_models.integrated_oda_estimate.ValidatingModel import SequenceValidatingModels
from python_choice_models.integrated_oda_estimate.Estimators import estimators


def run_with_soft(estimation_method, model, products, weight, in_sample_transactions, out_of_sample_transactions_prob):
    model_info = estimators[estimation_method]['models'][model]

    # print(' * Creating initial solution...')

    model = model_info['model_class'](products)
    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )


    result = model_info['estimator'].estimate(model, in_sample_transactions, weight)

    mae = result.mae_known_prob(out_of_sample_transactions_prob)

    return  mae

def Validating_Models_Soft(estimation_method, model, file_name,  K_sample, K_train, beta):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    N_prod = len(products)

    ##In sample: data for training
    # Data format: dict{"amount_products":X, "transactions":{{"in_sample"...}{"out_of_sample"..,}}
    # Each data piece {"products":xx."offered_products":[id of products]}
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    # print(in_sample_transactions)
    model_em = EmpiricalEstimator(in_sample_transactions)
    empirical_dict = model_em.empirical_dict
    Sequenced_validating_models, Sorted_deviation, Sorted_likelihood = SequenceValidatingModels(beta, N_prod,
                                                                                                K_sample - 1,
                                                                                                empirical_dict,
                                                                                                in_sample_transactions)
    K_sample_r = len(Sequenced_validating_models)
    WEIGHT_lst = [1, 10, 100, 1000]
    K_ae = [[] for s in range(K_sample_r)]
    for s in range(K_sample_r):
        for k in range(K_train):
            # results = []
            aes = []
            validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                       N_prod,
                                                                                       Sequenced_validating_models[s])

            for weight in WEIGHT_lst:
                mae = run_with_soft(estimation_method, model, products, weight, validating_transactions,
                                    validating_prob)
                aes.append(mae)
            K_ae[s].append(aes)

    best_model = dict()
    for s in range(K_sample_r):

        for k in range(K_train):
            mae_lst = K_ae[s][k]
            for idx in range(len(mae_lst)):

                weight = WEIGHT_lst[idx]
                s_ae = mae_lst[idx]
                if weight in best_model:
                    val1 = best_model[weight]['ae']
                    val1 += s_ae
                    best_model[weight].update({'ae': val1})
                else:
                    best_model.update({weight: {'ae': s_ae}})
                idx += 1


    min_ae = 100
    opt_weight = 0
    for key in best_model:
        val1 = best_model[key]['ae']
        if val1 < min_ae:
            opt_weight = key
            min_ae = val1

    return opt_weight

def Validation_soft(estimation_method, model, file_name, K_train):
    # print(' * Reading input file...')
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    N_prod = len(products)

    ##In sample: data for training
    # Data format: dict{"amount_products":X, "transactions":{{"in_sample"...}{"out_of_sample"..,}}
    # Each data piece {"products":xx."offered_products":[id of products]}
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    # print(in_sample_transactions)
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    ##Out of sample: data for testing
    # out_of_sample_transactions = Transaction.from_json(data['transactions']['out_of_sample'])
    ##Out of sample: choice probability of test data
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])
    model_em = EmpiricalEstimator(in_sample_transactions)
    empirical_dict = model_em.empirical_dict


    pool = Pool(mp.cpu_count() - 2)

    K_results = []
    K_method_name = []
    K_model_name = []
    K_rmse = []
    K_ae = []

    WEIGHT_lst = [1, 10, 100, 1000]
    for k in range(K_train):

        aes = []
        validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                   N_prod, empirical_dict)

        for weight in WEIGHT_lst:

            mae = run_with_soft(estimation_method, model, products, weight, validating_transactions, validating_prob)
            aes.append(mae)

        K_ae.append(aes)

    best_model = dict()
    for index in range(len(K_ae)):
        ae_lst = K_ae[index]
        for idx in range(len(ae_lst)):

            weight = WEIGHT_lst[idx]
            s_ae = ae_lst[idx]
            if weight in best_model:
                val1 = best_model[weight]['ae']
                val1 += s_ae
                best_model[weight].update({'ae': val1})
            else:
                best_model.update({weight: { 'ae': s_ae}})
            idx += 1

        index += 1
    min_ae = 100
    opt_weight = 0
    for key in best_model:
        val1 = best_model[key]['ae']
        if val1 < min_ae:
            opt_weight = key
            min_ae = val1

    return opt_weight


def Soft_Regularize(estimation_method, model,file_name, K_train):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()

    # opt_weight = Validation_soft(estimation_method, model,file_name, K_train)
    opt_weight = Validating_Models_Soft(estimation_method, model,file_name,  5, K_train,  0.02)
    model_name = model
    products = list(range(data['amount_products']))


    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])


    model_info = estimators[estimation_method]['models'][model_name]

    model = model_info['model_class'](products)

    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )

    result = model_info['estimator'].estimate(model, in_sample_transactions, opt_weight)
    error_dict = {}
    rmse_in = result.rmse_known_prob(in_sample_transactions_prob)
    ae_in = result.ae_known_prob(in_sample_transactions_prob)
    mrmse_in = result.mrmse_known_prob(in_sample_transactions_prob)
    mae_in = result.mae_known_prob(in_sample_transactions_prob)
    error_dict.update({"rmse_in": rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in": mae_in})

    rmse_out = result.rmse_known_prob(out_of_sample_transactions_prob)
    ae_out = result.ae_known_prob(out_of_sample_transactions_prob)
    mrmse_out = result.mrmse_known_prob(out_of_sample_transactions_prob)
    mae_out = result.mae_known_prob(out_of_sample_transactions_prob)
    error_dict.update({"rmse_out": rmse_out, "ae_out": ae_out, "mrmse_out": mrmse_out, "mae_out": mae_out})

    rmse_all = result.rmse_known_prob(all_sample_transactions_prob)
    ae_all = result.ae_known_prob(all_sample_transactions_prob)
    mrmse_all = result.mrmse_known_prob(all_sample_transactions_prob)
    mae_all = result.mae_known_prob(all_sample_transactions_prob)

    aic = result.aic_for(in_sample_transactions)
    bic = result.bic_for(in_sample_transactions)
    chi2 = result.hard_chi_squared_score_for(in_sample_transactions)
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})
    error_dict.update({"AIC": aic, "BIC": bic, "chi2": chi2})
    return error_dict
