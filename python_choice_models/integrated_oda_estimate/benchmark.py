# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import json
import random
from math import sqrt
import numpy as np


from python_choice_models.estimation.market_explore.ranked_list import MIPMarketExplorer
from python_choice_models.estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.latent_class import LatentClassExpectationMaximizationEstimator
from python_choice_models.estimation.maximum_likelihood.random_choice import RandomChoiceModelMaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from python_choice_models.estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator


from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator

from python_choice_models.settings import Settings

from python_choice_models.models import MixedLogitModel, MultinomialLogitModel, ExponomialModel, LatentClassModel, MarkovChainModel, MarkovChainRank2Model, NestedLogitModel, RandomChoiceModel, RankedListModel

GLOBAL_TIME_LIMIT = 1800

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

estimators = {
    'max': {
        'name': 'Standard Maximum Likelihood',
        'models': {
            'exp': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: ExponomialModel.simple_deterministic(products),
                'name': 'Exponomial',
                'settings': NORMAL_SETTINGS
            },
            'mkv': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MarkovChainModel.simple_deterministic(products),
                'name': 'Markov Chain',
                'settings': NORMAL_SETTINGS
            },
            'mkv2': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MarkovChainRank2Model.simple_deterministic(products),
                'name': 'Markov Chain Rank 2',
                'settings': NORMAL_SETTINGS
            },
            'mnl': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MultinomialLogitModel.simple_deterministic(products),
                'name': 'Multinomial Logit',
                'settings': NORMAL_SETTINGS
            },
            'nl': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: NestedLogitModel.simple_deterministic_ordered_nests(products, [1, len(products) // 2, len(products) - (len(products) // 2) - 1]),
                'name': 'Nested Logit',
                'settings': NORMAL_SETTINGS
            },
            'mx': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MixedLogitModel.simple_deterministic(products),
                'name': 'Mixed Logit',
                'settings': NORMAL_SETTINGS
            },
            'rnd': {
                'estimator': RandomChoiceModelMaximumLikelihoodEstimator(),
                'model_class': lambda products: RandomChoiceModel.simple_deterministic(products),
                'name': 'Random Choice',
                'settings': NORMAL_SETTINGS
            },
            'rl': {
                'estimator': RankedListMaximumLikelihoodEstimator.with_this(MIPMarketExplorer()),
                'model_class': lambda products: RankedListModel.simple_deterministic_independent(products),
                'name': 'Ranked List',
                'settings': RANKED_LIST_SETTINGS
            },
            'lc': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 5),
                'name': 'Latent Class',
                'settings': RANKED_LIST_SETTINGS
            },

        }
    },
    'em': {
        'name': 'Expectation Maximization',
        'models': {
            'mkv': {
                'estimator': MarkovChainExpectationMaximizationEstimator(),
                'model_class': lambda products: MarkovChainModel.simple_deterministic(products),
                'name': 'Markov Chain',
                'settings': NORMAL_SETTINGS
            },
            'rl': {
                'estimator': RankedListExpectationMaximizationEstimator.with_this(MIPMarketExplorer()),
                'model_class': lambda products: RankedListModel.simple_deterministic_independent(products),
                'name': 'Ranked List',
                'settings': RANKED_LIST_SETTINGS
            },
            'lc': {
                'estimator': LatentClassExpectationMaximizationEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 5),
                'name': 'Latent Class',
                'settings': RANKED_LIST_SETTINGS
            }
        }
    },
    'fw': {
        'name': 'Frank Wolfe/Conditional Gradient',
        'models': {
            'lc': {
                'estimator': LatentClassFrankWolfeEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 1),
                'name': 'Latent Class',
                'settings': NORMAL_SETTINGS
            },
        }
    }
}



#The best model that can generate if we know the true ground model
def SmoothingTrueModel(file_name):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    Ground_model = str(data['Ground'])


    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])

    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])


    alpha_pool = [0, 0.25, 0.5, 0.75, 0.9, 1]
    Estimation_method = ''
    model_em = EmpiricalEstimator(in_sample_transactions)
    empirical_dict = model_em.emprical_dict
    if Ground_model == 'lc':
        Estimation_method = 'fw'
    elif Ground_model == 'mkv':
        Estimation_method = 'em'
    elif Ground_model in ['mnl', 'rl', 'exp', 'emnl', 'nl', 'rnd']:
        Estimation_method = 'max'
    else:
        Ground_model = 'exp'
        Estimation_method = 'max'

    model_info = estimators[Estimation_method]['models'][Ground_model]
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

    rmse_in = []
    ae_in = []
    mrmse_in = []
    mae_in = []
    rmse_out = []
    ae_out = []
    mrmse_out = []
    mae_out = []
    rmse_all = []
    ae_all = []
    mrmse_all = []
    mae_all = []
    for alpha in alpha_pool:
        rmse_in.append(result.kernel_smooth_RMSE_known_prob(in_sample_transactions_prob, empirical_dict, alpha))
        rmse_out.append(result.kernel_smooth_RMSE_known_prob(out_of_sample_transactions_prob, empirical_dict, alpha))
        rmse_all.append(result.kernel_smooth_RMSE_known_prob(all_sample_transactions_prob, empirical_dict, alpha))

        mrmse_in.append(result.kernel_smooth_MRMSE_known_prob(in_sample_transactions_prob, empirical_dict, alpha))
        mrmse_out.append(result.kernel_smooth_MRMSE_known_prob(out_of_sample_transactions_prob, empirical_dict, alpha))
        mrmse_all.append(result.kernel_smooth_MRMSE_known_prob(all_sample_transactions_prob, empirical_dict, alpha))

        ae_in.append(result.kernel_smooth_AE_known_prob(in_sample_transactions_prob, empirical_dict, alpha))
        ae_out.append(result.kernel_smooth_AE_known_prob(out_of_sample_transactions_prob, empirical_dict, alpha))
        ae_all.append(result.kernel_smooth_AE_known_prob(all_sample_transactions_prob, empirical_dict, alpha))

        mae_in.append(result.kernel_smooth_MAE_known_prob(in_sample_transactions_prob, empirical_dict, alpha))
        mae_out.append(result.kernel_smooth_MAE_known_prob(out_of_sample_transactions_prob, empirical_dict, alpha))
        mae_all.append(result.kernel_smooth_MAE_known_prob(all_sample_transactions_prob, empirical_dict, alpha))
    # find the best one, return the results and corresponding alpha
    # print('Ae')
    # print(ae)
    error_dict = {}
    best_ae = min(mae_all)
    best_os_index = mae_all.index(best_ae)

    error_dict.update({"rmse_in": rmse_in[best_os_index], "ae_in": ae_in[best_os_index],
                       "mrmse_in": mrmse_in[best_os_index], "mae_in": mae_in[best_os_index]})
    error_dict.update({"rmse_out": rmse_out[best_os_index], "ae_out": ae_out[best_os_index],
                       "mrmse_out": mrmse_out[best_os_index], "mae_out": mae_out[best_os_index]})
    error_dict.update({"rmse_all": rmse_all[best_os_index], "ae_all": ae_all[best_os_index],
                       "mrmse_all": mrmse_all[best_os_index], "mae_all": mae_all[best_os_index]})
    # return rmse_in[best_os_index], ae_in[best_os_index], mrmse_in[best_os_index], mae_in[best_os_index],\
    #        rmse_out[best_os_index], ae_out[best_os_index], mrmse_out[best_os_index], mae_out[best_os_index],\
    #        rmse_all[best_os_index], ae_all[best_os_index], mrmse_all[best_os_index], mae_all[best_os_index],\
    #        Ground_model, alpha_pool[best_os_index]
    return error_dict



def RandomEstimationError(out_of_sample_transactions_prob):
    K_train = 10
    rmses = np.zeros(K_train)
    mrmses = np.zeros(K_train)
    aes = np.zeros(K_train)
    maes = np.zeros(K_train)
    for k in range(K_train):
        rmse = 0.0
        mrmse = 0.0
        ae = 0.0
        mae = 0.0
        amount_terms = 0
        amount_ae_terms = 0
        for transaction in out_of_sample_transactions_prob:
            num_product = len(transaction.offered_products)
            for probability_2 in transaction.prob:
                probability_1 = random.random()

                rmse += ((probability_1 - probability_2) ** 2)

                ae += abs(probability_1 - probability_2)
                if num_product > 1:
                    mrmse +=  ((probability_1 - probability_2) ** 2) * num_product / (num_product - 1)
                    mae += abs(probability_1 - probability_2) * 1/sqrt(num_product)
                else:
                    mrmse += ((probability_1 - probability_2) ** 2)
                    mae += abs(probability_1 - probability_2)
                amount_terms += 1
            amount_ae_terms += 1

        rmses[k] = sqrt(rmse / float(amount_terms))
        mrmses[k] = sqrt(mrmse/ float(amount_ae_terms))
        aes[k] = ae / float(amount_ae_terms)
        maes[k] = mae / float(amount_ae_terms)

    return np.average(rmses), np.average(aes), np.average(mrmses), np.average(maes)

## return the performance of a random choice model (the worst performance)
def RandomlyPickProb(file_name):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))
    Ground_model = str(data['Ground'])


    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])

    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])

    error_dict = {}
    rmse_in, ae_in, mrmse_in, mae_in = RandomEstimationError(in_sample_transactions_prob)
    error_dict.update({"rmse_in": rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in": mae_in})
    rmse_out, ae_out, mrmse_out, mae_out = RandomEstimationError(out_of_sample_transactions_prob)
    error_dict.update({"rmse_out": rmse_out, "ae_out": ae_out, "mrmse_out": mrmse_out, "mae_out": mae_out})
    rmse_all, ae_all, mrmse_all, mae_all = RandomEstimationError(all_sample_transactions_prob)
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})


    return error_dict