# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.estimation.expectation_maximization import ExpectationMaximizationEstimator
from python_choice_models.estimation.ranked_list import RankedListEstimator


class RankedListExpectationMaximizationEstimator(ExpectationMaximizationEstimator, RankedListEstimator):
    def one_step(self, model, transactions):
        x = [[0 for _ in transactions] for _ in model.ranked_lists]

        for t, transaction in enumerate(transactions):
            compatibles = model.ranked_lists_compatible_with(transaction)
            den = sum([model.beta_for(compatible[0]) for compatible in compatibles])
            for i, ranked_list in compatibles:
                x[i][t] = model.beta_for(i) / den

        m = [sum(x[i]) for i in range(len(model.ranked_lists))]
        model.set_betas([m[i] / sum(m) for i in range(len(model.ranked_lists))])
        return model
