# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
import json
import numpy as np
import time

from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator

from python_choice_models.settings import Settings

GLOBAL_TIME_LIMIT = 1800



## use bootstrap to generate the data for validation
def GenerateOutofSampleTransactions(in_sample_transactions, n, empirical_dict):
    #copy list
    # Validating_dict = ValidatingModel(empirical_dict, n, beta)
    for offered_set, value in empirical_dict.items():
        # del Validating_dict[offered_set]['count_assor']
        for product in offered_set:
            if product in empirical_dict[offered_set]:
                continue
            else:
                empirical_dict[offered_set].update({product: 0.0})



    out_of_sample_transactions = deepcopy(in_sample_transactions)
    out_of_sample_transactions_prob = []
    for transaction in out_of_sample_transactions:
        tmp_dict = {}
        offered_set = transaction.offered_products
        choice_prob = np.zeros(n + 1)
        offered_product_prob = []
        for product in offered_set:
            # print(Validating_dict[offered_set][product])
            # print(empirical_dict[tuple(offered_set)][product])
            if tuple(offered_set) in empirical_dict:
                if product in empirical_dict[tuple(offered_set)]:
                    choice_prob[product] = empirical_dict[tuple(offered_set)][product]
                    offered_product_prob.append(empirical_dict[tuple(offered_set)][product])


        choice = np.random.choice(n + 1, 1, p=choice_prob)
        transaction.product = choice[0]
        transaction_prob = Transaction_Extend(offered_product_prob, offered_set)
        # tmp_dict['prob'] = offered_product_prob
        # tmp_dict['Offered products'] = offered_set
        out_of_sample_transactions_prob.append(transaction_prob)

    return out_of_sample_transactions, out_of_sample_transactions_prob



def run_with_os(estimation_method, model, in_sample_transactions,
                out_of_sample_transactions_prob, products, estimators):
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


    mrmse = result.mrmse_known_prob(out_of_sample_transactions_prob)
    mae = result.mae_known_prob(out_of_sample_transactions_prob)


    return mrmse, mae


def Optimal_Operational_Statistics(file_name, N_prod, estimators, K_train):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))


    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    # out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])

    # print(in_sample_transactions)
    model_em = EmpiricalEstimator(in_sample_transactions)
    empirical_dict = model_em.empirical_dict

    pool = Pool(mp.cpu_count() - 2)

    K_results = []
    K_method_name = []
    K_model_name = []
    K_rmse = []
    K_ae = []


    for k in range(K_train):
        results = []
        rmses = []
        aes = []
        method_name = []
        model_name = []
        validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                   N_prod, empirical_dict)


        for estimation_method, method_info in list(estimators.items()):
            for model, model_info in list(method_info['models'].items()):
                # print('\tEST.\tMODEL\tH-RMSE\tH-AE\tS-RMSE.\tS-AE\tTIME')

                # print(estimation_method, model, alpha)
                # result = pool.apply_async(show_row_oda,args=(estimation_method, model, validating_transactions,
                #                                             validating_prob, products))
                    #record the result
                rmse_ground, ae_ground = run_with_os(estimation_method, model, validating_transactions,
                                                            validating_prob, products, estimators)
                method_name.append(estimation_method)
                model_name.append(model)
                # results.append(result)
                rmses.append(rmse_ground)
                aes.append(ae_ground)

        # K_results.append(results)
        K_rmse.append(rmses)
        K_ae.append(aes)
        K_model_name.append(model_name)
        K_method_name.append(method_name)


    # for results in K_results:
    #     for result in results:
    #         result.wait()
    # pool.close()
    # pool.join()

    min_S_RMSE = 100.0
    min_S_AE = 100.0
    best_S_RMSE_model = ''
    best_S_RMSE_alpha = 0.0
    best_model = dict()

    # print(K_model_name)
    # print(K_method_name)
    # print(K_kernel_coeff)


    # index = 0
    # for k_result in K_results:
    for index in range(len(K_ae)):
        model_name = K_model_name[index]
        method_name = K_method_name[index]
        rmse_lst = K_rmse[index]
        ae_lst = K_ae[index]
        # idx = 0
        for idx in range(len(ae_lst)):
        # for result in k_result:
            # error = result.get()
            # s_rmse = error[0]
            # s_ae = error[1]
            s_rmse = rmse_lst[idx]
            s_ae = ae_lst[idx]
            model = model_name[idx]
            method = method_name[idx]

            if model in best_model:
                if method in best_model[model]:
                    val1 = best_model[model][method]['rmse']
                    val1 += s_rmse
                    best_model[model][method].update({'rmse': val1})

                    val2 = best_model[model][method]['ae']
                    val2 += s_ae
                    best_model[model][method].update({'ae': val2})
                else:
                    best_model[model].update({method:{'rmse': s_rmse, 'ae': s_ae}})
            else:
                best_model.update({model:{method: {'rmse': s_rmse, 'ae': s_ae}}})
            idx += 1

        index += 1


    min_rmse = 100
    optimal_model = ''
    optimal_method = ''
    # for key_model in best_model:
    #     val1 = best_model[key_model]['ae']
    #     if val1 < min_ae:
    #         optimal_model = key_model
    #         min_ae = val1
    # use the modified rmse as the measurement in selecting the best structural model
    for key_model in best_model:
        for key_method in best_model[key_model]:
            val1 = best_model[key_model][key_method]['rmse']
            if val1 < min_rmse:
                optimal_model = key_model
                optimal_method = key_method
                min_rmse = val1
    # print(best_model)

    return optimal_model, optimal_method

#validate the prediction error of chosen optimal

def Validate_Bootstrap(file_name, N_prod,  K_train, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])


    t1 = time.time()


    optimal_model, optimal_method = Optimal_Operational_Statistics(file_name, N_prod, estimators, K_train)

    model_name = optimal_model
    outline = ['%s' % optimal_model]
    # bootstrap_file.write(','.join(outline) + '\n')
    products = list(range(data['amount_products']))

    # print(optimal_model, optimal_method, model_name)
    # print(estimators)
    model_info = estimators[optimal_method]['models'][model_name]

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
    t2 = time.time()
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
    error_dict.update({"time": t2 - t1})
    return error_dict


def Validate_Model_str(file_name, optimal_model, optimal_method, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])



    # optimal_model = Optimal_Operational_Statistics(file_name, N_prod, estimators, K_train)

    model_name = optimal_model
    products = list(range(data['amount_products']))

    model_info = estimators[optimal_method]['models'][model_name]

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


