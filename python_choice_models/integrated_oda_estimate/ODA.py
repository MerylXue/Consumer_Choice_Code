# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.



import json
import time
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
import numpy as np
from collections import Counter



from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator
from python_choice_models.settings import Settings

from python_choice_models.integrated_oda_estimate.ValidatingModel import GenerateOutofSampleTransactions, SequenceValidatingModels, SampleTrainingTransactions
from python_choice_models.integrated_oda_estimate.Estimators import estimators
GLOBAL_TIME_LIMIT = 1800



def run_with_os(estimation_method, model, in_sample_transactions,
                out_of_sample_transactions_prob, products, alpha, estimators):

    model_info = estimators[estimation_method]['models'][model]

    model = model_info['model_class'](products)
    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )

    if hasattr(model_info['estimator'], 'estimate_with_market_discovery'):
        result = model_info['estimator'].estimate_with_market_discovery(model, in_sample_transactions)
    else:
        result = model_info['estimator'].estimate(model, in_sample_transactions)


    if alpha == 1:
        mrmse = result.rmse_known_prob(out_of_sample_transactions_prob)
        mae = result.ae_known_prob(out_of_sample_transactions_prob)
    elif alpha < 1:
        count_dict = EmpiricalEstimator(in_sample_transactions).emprical_dict
        mrmse = result.kernel_smooth_MRMSE_known_prob(out_of_sample_transactions_prob, count_dict, alpha)
        mae = result.kernel_smooth_MAE_known_prob(out_of_sample_transactions_prob, count_dict, alpha)

    return mrmse, mae

# choose the best structural operational statistics and the value for alpha
def Optimal_Operational_Statistics(file_name, beta, K_sample, K_train, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    N_prod = len(products) - 1

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])

    empirical_dict = EmpiricalEstimator(in_sample_transactions).emprical_dict

    pool = Pool(mp.cpu_count() - 2)
    Sequenced_validating_models, Sorted_deviation, Sorted_likelihood = SequenceValidatingModels(beta, N_prod,
                                                                                                K_sample - 1,
                                                                                                empirical_dict,
                                                                                                in_sample_transactions)
    ## grid search for the values of alpha
    alpha_pool = [0, 0.25, 0.5, 0.75, 0.9, 1]

    K_sample_r = len(Sequenced_validating_models)
    K_result_mrmse = [[] for s in range(K_sample_r)]
    K_result_mae = [[] for s in range(K_sample_r)]
    K_method_name = [[] for s in range(K_sample_r)]
    K_model_name = [[] for s in range(K_sample_r)]
    K_kernel_coeff = [[] for s in range(K_sample_r)]

    for s in range(K_sample_r):
        for k in range(K_train):
            maes = []
            mrmses = []
            method_name = []
            model_name = []
            kernel_coeff = []
            validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                               N_prod, Sequenced_validating_models[s])
            for estimation_method, method_info in list(estimators.items()):
                for model, model_info in list(method_info['models'].items()):
                    for alpha in alpha_pool:
                        # result = pool.apply_async(run_with_os,args=(estimation_method, model, validating_transactions,
                        #                                             validating_prob, products, alpha, estimators))
                        mrmse, mae = run_with_os(estimation_method, model, validating_transactions,
                                                                    validating_prob, products, alpha, estimators)
                        #record the result
                        method_name.append(estimation_method)
                        model_name.append(model)
                        kernel_coeff.append(alpha)
                        # results.append(result)
                        maes.append(mae)
                        mrmses.append(mrmse)

            # K_results[s].append(results)
            K_result_mrmse[s].append(mrmses)
            K_result_mae[s].append(maes)
            K_method_name[s].append(method_name)
            K_model_name[s].append(model_name)
            K_kernel_coeff[s].append(kernel_coeff)



    best_model = [dict() for s in range(K_sample)]

    #find the best os for each sampled validating model
    for s in range(K_sample):
        # index = 0
        for k in range(K_train):
            mrmse_lst = K_result_mrmse[s][k]
            mae_lst = K_result_mae[s][k]
            method_name = K_method_name[s][k]
            model_name = K_model_name[s][k]
            kernel_coeff = K_kernel_coeff[s][k]
            for idx in range(len(mae_lst)):
                s_mrmse = mrmse_lst[idx]
                s_mae = mae_lst[idx]
                method = method_name[idx]
                model = model_name[idx]


                alpha = kernel_coeff[idx]

                if model in best_model[s]:
                    if alpha in best_model[s][model]:
                        if method in best_model[s][model][alpha]:
                            val1 = best_model[s][model][alpha][method]['rmse']
                            val1 += s_mrmse
                            best_model[s][model][alpha][method].update({'rmse': val1})

                            val2 = best_model[s][model][alpha][method]['ae']
                            val2 += s_mae
                            best_model[s][model][alpha][method].update({'ae': val2})
                        else:
                            best_model[s][model][alpha].update({method:{'rmse': s_mrmse, 'ae': s_mae}})
                    else:
                        best_model[s][model].update({alpha:{method:{'rmse': s_mrmse, 'ae': s_mae}}})
                else:
                    best_model[s].update({model:{alpha:{method:{'rmse': s_mrmse, 'ae': s_mae}}}})


    optimal_model = ['' for s in range(K_sample_r)]
    optimal_alpha = [0.0 for s in range(K_sample_r)]
    optimal_method = ['' for s in range(K_sample_r)]
    for s in range(K_sample):
        min_ae = 100
        for key_model in best_model[s]:
            for key_alpha in best_model[s][key_model]:
                for key_method in best_model[s][key_model][key_alpha]:
                    val1 = best_model[s][key_model][key_alpha][key_method]['ae']/K_train
                    if val1 < min_ae:
                        optimal_model[s] = key_model
                        optimal_alpha[s] = key_alpha
                        optimal_method[s] = key_method
                        min_ae = val1


    rmse_table = np.zeros((K_sample, K_sample))
    ae_table = np.zeros((K_sample, K_sample))

    for k in range(K_sample):  # validating model (as row)
        for s in range(K_sample):  # optimal os (as col)
            val1 = best_model[k][optimal_model[s]][optimal_alpha[s]][optimal_method[s]]['rmse']/K_train
            val2 = best_model[k][optimal_model[s]][optimal_alpha[s]][optimal_method[s]]['ae']/K_train
            rmse_table[k][s] = val1
            ae_table[k][s] = val2


    return optimal_model, optimal_alpha, optimal_method, rmse_table, ae_table, Sequenced_validating_models, \
           Sorted_deviation, Sorted_likelihood



def Max_Regret_OS(file_name,  beta, K_sample, K_train, estimators):
    # K_sample = 10

    optimal_model, optimal_alpha, optimal_method, rmse_table, ae_table, Sequenced_validating_models, Sorted_deviation, Sorted_likelihood = Optimal_Operational_Statistics(
        file_name, beta, K_sample, K_train, estimators)
    # k * k submatrix

    # calculate the regret
    regret_error = np.zeros((K_sample, K_sample))

    for k in range(K_sample):
        min_regret = min(ae_table[k]) #min_error(model, os) for all os
        for s in range(K_sample):
            regret_error[k][s] = ae_table[k][s] - min_regret
    #calcalte the maximum regret
    max_regret_error = np.zeros((K_sample, K_sample)) # for each os, find the max

    for s in range(K_sample):
        for k in range(K_sample):
            sub_table = regret_error[0:k + 1,s]
            max_regret_error[k][s] = np.max(sub_table)


    return Sorted_deviation, Sorted_likelihood, max_regret_error, optimal_model, optimal_alpha, optimal_method, regret_error, ae_table


## pick the optimal operational statistics by min-max regeret

def Min_Max_Regret_OS( K_sample, max_regret_error):
    min_max_regret_os = np.zeros(K_sample)
    for k in range(K_sample):
        min_max_regret = min(max_regret_error[k])
        min_max_regret_os[k] = list(max_regret_error[k]).index(min_max_regret)

    d2 = Counter(min_max_regret_os)
    sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)

    optimal_os_index = int(sorted_x[0][0])

    return optimal_os_index


def Validate_ODA(file_name, optimal_model, optimal_method, optimal_alpha, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    model_name = optimal_model
    products = list(range(data['amount_products']))

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])


    model_info = estimators[optimal_method]['models'][model_name]

    model = model_info['model_class'](products)

    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )
    # in_sample_transactions = SampleTrainingTransactions(in_sample_transactions_prob)

    if hasattr(model_info['estimator'], 'estimate_with_market_discovery'):
        result = model_info['estimator'].estimate_with_market_discovery(model, in_sample_transactions)
    else:
        result = model_info['estimator'].estimate(model, in_sample_transactions)

    count_dict = EmpiricalEstimator(in_sample_transactions).emprical_dict
    # optimal alpha dict is not null
    if optimal_alpha:
        rmse_in = result.kernel_smooth_RMSE_known_prob(in_sample_transactions_prob, count_dict, optimal_alpha)
        rmse_out = result.kernel_smooth_RMSE_known_prob(out_of_sample_transactions_prob, count_dict,
                                                             optimal_alpha)
        rmse_all = result.kernel_smooth_RMSE_known_prob(all_sample_transactions_prob, count_dict,
                                                             optimal_alpha)

        mrmse_in = result.kernel_smooth_MRMSE_known_prob(in_sample_transactions_prob, count_dict,
                                                              optimal_alpha)
        mrmse_out = result.kernel_smooth_MRMSE_known_prob(out_of_sample_transactions_prob, count_dict,
                                                               optimal_alpha)
        mrmse_all = result.kernel_smooth_MRMSE_known_prob(all_sample_transactions_prob, count_dict,
                                                               optimal_alpha)

        ae_in = result.kernel_smooth_AE_known_prob(in_sample_transactions_prob, count_dict, optimal_alpha)
        ae_out = result.kernel_smooth_AE_known_prob(out_of_sample_transactions_prob, count_dict,
                                                         optimal_alpha)
        ae_all = result.kernel_smooth_AE_known_prob(all_sample_transactions_prob, count_dict, optimal_alpha)

        mae_in = result.kernel_smooth_MAE_known_prob(in_sample_transactions_prob, count_dict, optimal_alpha)
        mae_out = result.kernel_smooth_MAE_known_prob(out_of_sample_transactions_prob, count_dict,
                                                           optimal_alpha)
        mae_all = result.kernel_smooth_MAE_known_prob(all_sample_transactions_prob, count_dict, optimal_alpha)
    else:
        print("Optimal alpha idct is null!!!!")
        rmse_in = result.rmse_known_prob(in_sample_transactions_prob)
        rmse_out = result.rmse_known_prob(out_of_sample_transactions_prob)
        rmse_all = result.rmse_known_prob(all_sample_transactions_prob)

        mrmse_in = result.mrmse_known_prob(in_sample_transactions_prob)
        mrmse_out = result.mrmse_known_prob(out_of_sample_transactions_prob)
        mrmse_all = result.mrmse_known_prob(all_sample_transactions_prob)

        ae_in = result.ae_known_prob(in_sample_transactions_prob)
        ae_out = result.ae_known_prob(out_of_sample_transactions_prob)
        ae_all = result.ae_known_prob(all_sample_transactions_prob)

        mae_in = result.mae_known_prob(in_sample_transactions_prob)
        mae_out = result.mae_known_prob(out_of_sample_transactions_prob)
        mae_all = result.mae_known_prob(all_sample_transactions_prob)

    error_dict = {}
    error_dict.update({"rmse_in": rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in": mae_in})
    error_dict.update({"rmse_out": rmse_out, "ae_out": ae_out, "mrmse_out": mrmse_out, "mae_out": mae_out})
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})
    return error_dict



def optimal_theoretical_os(file_name, optimal_models, optimal_methods, optimal_alphas):
    rmse_table_in = np.zeros(len(optimal_models))
    ae_table_in = np.zeros(len(optimal_models))
    rmse_table = np.zeros(len(optimal_models))
    ae_table = np.zeros(len(optimal_models))
    for k in range(len(optimal_models)):
        rmse_in, ae_in, rmse_all, ae_all = Validate_ODA(file_name, optimal_models[k], optimal_methods[k], optimal_alphas[k])
        rmse_table_in[k] = rmse_in
        rmse_table[k] = rmse_all
        ae_table_in[k] = ae_in
        ae_table[k] = ae_all

    #find the best one
    min_ae = min(ae_table)
    optimal_os_index = list(ae_table).index(min_ae)

    return optimal_os_index, rmse_table_in, ae_table_in, rmse_table, ae_table


def Validate_MinMaxRegret_OS(input_file, beta, K_sample, K_train_os, estimators):

    Sorted_deviation, Sorted_likelihood, max_regret_error, optimal_model, optimal_alpha, optimal_method, regret_error, ae_table = Max_Regret_OS(
        input_file,  beta,
        K_sample, K_train_os, estimators)
    # optimal os chosen by min max regret
    optimal_os_regret = Min_Max_Regret_OS(K_sample, max_regret_error)
    error_dict = Validate_ODA(input_file, optimal_model[optimal_os_regret], optimal_method[optimal_os_regret],
                            optimal_alpha[optimal_os_regret], estimators)
    return error_dict


