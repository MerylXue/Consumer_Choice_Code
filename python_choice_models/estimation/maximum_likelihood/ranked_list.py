# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.ranked_list import RankedListEstimator


class RankedListMaximumLikelihoodEstimator(MaximumLikelihoodEstimator, RankedListEstimator):
    pass
