# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import json
import numpy as np
from collections import Counter
from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator
from python_choice_models.settings import Settings


from python_choice_models.integrated_oda_estimate.ValidatingModel import GenerateOutofSampleTransactions, SequenceValidatingModels, SampleTrainingTransactions
from python_choice_models.integrated_oda_estimate.SmoothingParameter import get_alpha
from python_choice_models.optimization.regret_optimization import Min_Max_Regret_OS
from python_choice_models.data_generate.Data_transaction_Multiple_sample import DataProbTransferJson

GLOBAL_TIME_LIMIT = 1800

def run_with_os_alpha(estimation_method, model, in_sample_transactions,
                out_of_sample_transactions_prob, products, alpha_dict, estimators):

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



    count_dict = model.EmpiricalEstimation(in_sample_transactions)
    # rmse = result.kernel_smooth_MRMSE_assortment_dict(out_of_sample_transactions_prob, count_dict, alpha_dict)
    # ae = result.kernel_smooth_MAE_assortment_dict(out_of_sample_transactions_prob, count_dict, alpha_dict)

    rmse = result.kernel_smooth_MRMSE_assortment_dict(out_of_sample_transactions_prob, count_dict, alpha_dict)
    ae = result.kernel_smooth_MAE_assortment_dict(out_of_sample_transactions_prob, count_dict, alpha_dict)
    return rmse, ae

def Generate_os_family(in_sample_transactions, Sequenced_validating_models, products, estimators):
    N_prod = len(products) - 1
    os_family = []
    K_sample_r = len(Sequenced_validating_models)

    for s in range(K_sample_r):
        validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                   N_prod,
                                                                                   Sequenced_validating_models[s])
        os_dict = {}
        for estimation_method, method_info in list(estimators.items()):
            for model, model_info in list(method_info['models'].items()):
                # os_family.update({model: estimation_method})
                alpha_dict = get_alpha(estimation_method, model, products, validating_transactions, validating_prob)
                # print(alpha_dict)
                alpha_assor = list(alpha_dict.keys())
                alpha_list = list(alpha_dict.values())
                # os_dict['model'] = model
                # os_dict['estimation_method'] = estimation_method
                os_dict.update({tuple([model, estimation_method]): {'assor': alpha_assor, 'alpha': alpha_list}})

                # os_family.update({tuple([model, estimation_method]): {'alpha': alpha_list}})
                # for key_assor, value in alpha_dict.items():
                    # os_family.update({model:{estimation_method: {key_assor: value}}})
                # print(os_family[tuple([model, estimation_method])])
        # print("Validaing -------%d------------"%s)
        # print(os_dict)
        os_family.append(os_dict)
    # print("os_family_%d"%len(os_family))
    # print(os_family)
    return os_family



def Optimal_Operational_Statistics(file_name, beta, K_sample, K_train, estimators):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    N_prod = len(products) - 1

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    empirical_dict = EmpiricalEstimator(in_sample_transactions).emprical_dict
    Sequenced_validating_models, Sorted_deviation, Sorted_likelihood = SequenceValidatingModels(beta, N_prod,
                                                                                                K_sample - 1,
                                                                                                empirical_dict,
                                                                                                in_sample_transactions)
    K_sample_r = len(Sequenced_validating_models)
    os_dict = Generate_os_family(in_sample_transactions, Sequenced_validating_models, products, estimators)

    # print(os_family)
    # pool = Pool(mp.cpu_count() - 2)
    # print("Os family")
    # print(os_family)
    # K_results = [[] for s in range(K_sample_r)]
    K_mrmse = [[] for s in range(K_sample_r)]
    K_mae = [[] for s in range(K_sample_r)]
    K_model = [[] for s in range(K_sample_r)]
    K_method = [[] for s in range(K_sample_r)]
    for s in range(K_sample_r):
        os_family = os_dict[s]
        for k in range(K_train):
            # results = []
            models = []
            methods = []
            mrmses = []
            maes = []
            validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                       N_prod,
                                                                                       Sequenced_validating_models[s])

            for key in os_family:
                # model = os_dict['model']
                # estimation_method = os_dict['estimation_method']

                    # for key in os_family[model][estimation_method]:
                    #     val = os_family[model][estimation_method][key]
                    #     alpha_dict.update({key, val})
                alpha_assor = os_family[key]['assor']
                alpha_list = os_family[key]['alpha']
                # print("Alpha_dict-------------------------")
                alpha_dict = dict(zip(alpha_assor, alpha_list))
                # print(alpha_dict)
                model = list(key)[0]
                estimation_method = list(key)[1]
                # print(key, model, estimation_method)
                mrmse, mae = run_with_os_alpha(estimation_method, model, validating_transactions,validating_prob,
                                               products, alpha_dict, estimators)
                # result = pool.apply_async(run_with_os_alpha ,args=(estimation_method, model, validating_transactions,
                #                   validating_prob, products, alpha_dict))
                # res   ults.append(result)
                mrmses.append(mrmse)
                maes.append(mae)
                models.append(model)
                methods.append(estimation_method)
            K_mrmse[s].append(mrmses)
            K_mae[s].append(maes)
            # K_results[s].append(results)
            K_model[s].append(models)
            K_method[s].append(methods)



    best_model = [dict() for s in range(K_sample)]
    # find the best os for each sampled validating model
    for s in range(K_sample_r):
        for k in range(K_train):
            mrmse_lst = K_mrmse[s][k]
            mae_lst = K_mae[s][k]
            method_name = K_method[s][k]
            model_name = K_model[s][k]
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
            #
            #     idx += 1
            # index += 1

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

    optimal_alpha_dicts = []
    for s in range(K_sample_r):
        os_family = os_dict[s]
        if tuple([optimal_model[s], optimal_method[s]]) in os_family:
            alpha_assor = os_family[tuple([optimal_model[s], optimal_method[s] ])]['assor']
            alpha_list = os_family[tuple([optimal_model[s], optimal_method[s]])]['alpha']
            alpha_dict = dict(zip(alpha_assor, alpha_list))
            optimal_alpha_dicts.append(alpha_dict)
        else:
            print("Optimal model and method not in os family!!!")

    return optimal_model, optimal_method, optimal_alpha_dicts, mrmse_table, mae_table, Sequenced_validating_models, \
           Sorted_deviation, Sorted_likelihood


def Max_Regret_OS_alpha_dict(file_name, beta, K_sample, K_train, estimators):
    # K_sample = 10
    print("start max regret os alpha dict")
    optimal_model, optimal_method, optimal_alpha_dicts, rmse_table, ae_table, Sequenced_validating_models, \
    Sorted_deviation, Sorted_likelihood = Optimal_Operational_Statistics(file_name,  beta, K_sample, K_train, estimators)
    # k * k submatrix
    # min_max_ae = np.zeros(K_sample)
    # min_regret_os_model = ['' for i in range(K_sample)]
    # min_regret_os_alpha = [0.0 for i in range(K_sample)]
    # calculate the regret
    regret_error = np.zeros((K_sample, K_sample))
    # print('ae')
    # print(ae_table)
    for k in range(K_sample):
        min_regret = min(ae_table[k]) #min_error(model, os) for all os
        for s in range(K_sample):
            regret_error[k][s] = ae_table[k][s] - min_regret
    # print('regret')
    # print(regret_error)
    #calcalte the maximum regret
    max_regret_error = np.zeros((K_sample, K_sample)) # for each os, find the max

    for s in range(K_sample):
        for k in range(K_sample):
            sub_table = regret_error[0:k + 1,s]
            # print(sub_table)
            max_regret_error[k][s] = np.max(sub_table)

    return Sorted_deviation, Sorted_likelihood, max_regret_error, optimal_model, optimal_method, optimal_alpha_dicts, regret_error, ae_table




def Validate_ODA_alpha_dict(file_name, optimal_model, optimal_method, optimal_alpha_dict, estimators):
    # print("Validate model-------------------")
    # print(optimal_model, optimal_method)
    # print(optimal_alpha_dict)
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
# print(optimal_model, optimal_alpha, optimal_method)
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




    count_dict = EmpiricalEstimator(in_sample_transactions).empirical_dict
    # optimal alpha dict is not null
    if optimal_alpha_dict:
        rmse_in = result.kernel_smooth_RMSE_assortment_dict(in_sample_transactions_prob, count_dict, optimal_alpha_dict)
        rmse_out = result.kernel_smooth_RMSE_assortment_dict(out_of_sample_transactions_prob, count_dict, optimal_alpha_dict)
        rmse_all = result.kernel_smooth_RMSE_assortment_dict(all_sample_transactions_prob, count_dict, optimal_alpha_dict)

        mrmse_in = result.kernel_smooth_MRMSE_assortment_dict(in_sample_transactions_prob, count_dict, optimal_alpha_dict)
        mrmse_out = result.kernel_smooth_MRMSE_assortment_dict(out_of_sample_transactions_prob, count_dict, optimal_alpha_dict)
        mrmse_all = result.kernel_smooth_MRMSE_assortment_dict(all_sample_transactions_prob, count_dict, optimal_alpha_dict)

        ae_in = result.kernel_smooth_AE_assortment_dict(in_sample_transactions_prob, count_dict, optimal_alpha_dict)
        ae_out = result.kernel_smooth_AE_assortment_dict(out_of_sample_transactions_prob, count_dict, optimal_alpha_dict)
        ae_all = result.kernel_smooth_AE_assortment_dict(all_sample_transactions_prob, count_dict, optimal_alpha_dict)

        mae_in = result.kernel_smooth_MAE_assortment_dict(in_sample_transactions_prob, count_dict, optimal_alpha_dict)
        mae_out = result.kernel_smooth_MAE_assortment_dict(out_of_sample_transactions_prob, count_dict, optimal_alpha_dict)
        mae_all = result.kernel_smooth_MAE_assortment_dict(all_sample_transactions_prob, count_dict, optimal_alpha_dict)


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

    aic_oda = result.AIC_kernel(in_sample_transactions, count_dict, optimal_alpha_dict)
    bic_oda = result.BIC_kernel(in_sample_transactions, count_dict, optimal_alpha_dict)
    chi2 = result.chi_square_kenerl(in_sample_transactions, count_dict, optimal_alpha_dict)
    error_dict = {}
    error_dict.update({"rmse_in": rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in": mae_in})
    error_dict.update({"rmse_out": rmse_out, "ae_out": ae_out, "mrmse_out": mrmse_out, "mae_out": mae_out})
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})
    error_dict.update({"AIC": aic_oda, "BIC": bic_oda, "chi2": chi2})
    return error_dict



## ODA method with the interpolation coefficient alpha(S) defined for each choice set
def Validate_MinMaxRegret_OS_alphadict(input_file, beta, K_sample, K_train_os, estimators):
    Sorted_deviation, Sorted_likelihood, max_regret_error, optimal_model, optimal_method, optimal_alpha_dicts, \
    regret_error, ae_table = Max_Regret_OS_alpha_dict(input_file, beta, K_sample, K_train_os, estimators)

    # optimal os chosen by min max regret
    optimal_os_regret = Min_Max_Regret_OS(K_sample, max_regret_error)



    error_dict = Validate_ODA_alpha_dict(input_file, optimal_model[optimal_os_regret],
                                                                                 optimal_method[optimal_os_regret],
                                                                                 optimal_alpha_dicts[optimal_os_regret],
                                         estimators)


    return error_dict, max(Sorted_deviation), optimal_model[optimal_os_regret], optimal_alpha_dicts[optimal_os_regret]

