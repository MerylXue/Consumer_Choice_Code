# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.integrated_oda_estimate.ValidatingModel import GenerateOutofSampleTransactions, SequenceValidatingModels, SampleTrainingTransactions
from python_choice_models.integrated_oda_estimate.SmoothingParameter import AverageErrorBound, CalculateAlpha


def get_alpha_multiple(estimation_method, model, products, in_sample_transactions, Sequenced_validating_models):
    N_prod = len(products) - 1
    sum_avg_sse = {}
    K_train = 10
    for s in range(len(Sequenced_validating_models)):
        for k in range(K_train):
            validating_transactions, validating_prob = GenerateOutofSampleTransactions(in_sample_transactions,
                                                                                       N_prod,
                                                                                       Sequenced_validating_models[s])
            avg_sse = AverageErrorBound(estimation_method, model, products, validating_transactions, validating_prob)
            for key_assor in avg_sse:
                if key_assor in sum_avg_sse:
                    for key_e, value in avg_sse[key_assor].items():
                        if key_e in sum_avg_sse:
                            val = sum_avg_sse[key_assor][key_e]
                            val += value/K_train
                            sum_avg_sse[key_assor].update({key_e: val})
                        else:
                            sum_avg_sse[key_assor].update({key_e: value/K_train})
                else:
                    sum_avg_sse.update({key_assor: {}})
                    for key_e, value in avg_sse[key_assor].items():
                        sum_avg_sse[key_assor].update({key_e: value/K_train})

    # average
    for key_assor in sum_avg_sse:
        for key_e, value in sum_avg_sse[key_assor].items():
            val = value/len(Sequenced_validating_models)
            sum_avg_sse[key_assor].update({key_e: val})

    alpha_dict = CalculateAlpha(estimation_method, model, products, sum_avg_sse, in_sample_transactions)
    # print('alpha_dict--------------------')
    # print(alpha_dict)
    return alpha_dict