## This code is from the paper
## Chen and Misic (2021), Decision Forest: A Nonparametric Approach to Modeling Irrational Choice
## https://ssrn.com/abstract=3376273
## The python code is written based on their code in Julia


import json
import time

from python_choice_models.decision_forest.decision_forest_estimate import DecisionForestEMEstimator
from python_choice_models.decision_forest import DecisionForestModel
from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.utils import NORMAL_SETTINGS
from python_choice_models.settings import Settings


#regulator = 1, all ones vector;
# = 0, zero vector;
# = e, 1/|S|+1
# = t, true probability in out-of-sample assortments




def runDecisionForest(file_name):
    # print(' * Reading input file...')
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))

    ##In sample: data for training
    # Data format: dict{"amount_products":X, "transactions":{{"in_sample"...}{"out_of_sample"..,}}
    # Each data piece {"products":xx."offered_products":[id of products]}
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])


    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])

    model = DecisionForestModel(products)
    Settings.new(
        linear_solver_partial_time_limit=NORMAL_SETTINGS['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=NORMAL_SETTINGS['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=NORMAL_SETTINGS['solver_total_time_limit'],
    )
    t1 = time.time()
    result, n_iter = DecisionForestEMEstimator().estimate(model, in_sample_transactions)
    # print(result)
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

    error_dict.update({"time":t2 - t1})
    error_dict.update({'num_iter': n_iter})
    #
    # dist_centroid = CalculateAdvDistance(model.index_dict, result.probabilities, list(result.in_sample_assort))
    # dist_true = CalculateAdvDistance(model.index_dict,true_prob, list(result.in_sample_assort))


    return error_dict

