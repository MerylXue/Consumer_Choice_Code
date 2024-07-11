# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import json
import os
import sys
import time
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/python_choice_models/')


from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.utils import update_error_by_dict_lst, add_details_to_dict
from python_choice_models.settings import Settings


GLOBAL_TIME_LIMIT = 1800

def run(estimation_method, model, file_name, estimators):

    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])

    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])

    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])


    input_file.close()
    products = list(range(data['amount_products']))
    t1 = time.time()
    # print(products)

    # print('    - Amount of transactions: %s' % len(data))
    # print('    - Amount of products: %s' % len(products))
    #
    # print(' * Retrieving estimation method...')
    model_info = estimators[estimation_method]['models'][model]

    # print(' * Retrieving settings...')
    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )
    print(Settings.instance().non_linear_solver_partial_time_limit())

    # print(' * Creating initial solution...')
    model = model_info['model_class'](products)

    # print(' * Starting estimation...')
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
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})

    aic = result.aic_for(in_sample_transactions)
    bic = result.bic_for(in_sample_transactions)
    chi2 = result.hard_chi_squared_score_for(in_sample_transactions)
    error_dict.update({"AIC": aic, "BIC": bic, "chi2": chi2})
    error_dict.update({"time": t2 - t1})

    return error_dict


## run all the estimation methods
def compare_all(file_name,estimators, result_dict, detail_dict):
    print('\tEST.\tMODEL\tH-RMSE\tH-AE\tTIME')
    pool = Pool(mp.cpu_count() - 2)
    results = []

    methods = []
    models = []


    for estimation_method, method_info in list(estimators.items()):
        for model, model_info in list(method_info['models'].items()):
            print(estimation_method, model)
            error_dict = run(estimation_method, model, file_name,estimators)
            results.append(pool.apply_async(run, args=(estimation_method, model, file_name)))
            # result_dict = update_error_by_dict_lst(model, estimation_method, error_dict, result_dict)
            methods.append(estimation_method)
            models.append(model)

    for result in results:
        result.wait()
    pool.close()
    pool.join()


    idx = 0
    for result in results:
        # try:
        error_dict = result.get()
        model = models[idx]
        method = methods[idx]
        idx += 1
        result_dict = update_error_by_dict_lst(model, method, error_dict, result_dict)
        detail_dict = add_details_to_dict(model, method, error_dict, detail_dict)

    return result_dict,detail_dict
