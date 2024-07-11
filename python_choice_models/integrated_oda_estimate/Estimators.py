# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

## This file defines the model and the estimation method for use

from python_choice_models.estimation.market_explore.ranked_list import MIPMarketExplorer
from python_choice_models.estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.latent_class import LatentClassExpectationMaximizationEstimator

from python_choice_models.estimation.maximum_likelihood.random_choice import RandomChoiceModelMaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from python_choice_models.estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator



from python_choice_models.models import Model, MixedLogitModel, MultinomialLogitModel, ExponomialModel, LatentClassModel, \
    MarkovChainModel, MarkovChainRank2Model, NestedLogitModel, RandomChoiceModel, RankedListModel



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
                'model_class': lambda products: NestedLogitModel.simple_deterministic_ordered_nests(products, [1,
                                                                                                               len(products) // 2,
                                                                                                               len(products) - (
                                                                                                                           len(products) // 2) - 1]),
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
            }
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

oda_estimators = {
    'max': {
        'name': 'Standard Maximum Likelihood',
        'models': {
            'exp': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: ExponomialModel.simple_deterministic(products),
                'name': 'Exponomial',
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
                'model_class': lambda products: NestedLogitModel.simple_deterministic_ordered_nests(products, [1,
                                                                                                               len(products) // 2,
                                                                                                               len(products) - (
                                                                                                                           len(products) // 2) - 1]),
                'name': 'Nested Logit',
                'settings': NORMAL_SETTINGS
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

