# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.



from python_choice_models.integrated_oda_estimate.empirical import EmpiricalEstimator

from python_choice_models.settings import Settings
from python_choice_models.integrated_oda_estimate.Estimators import oda_estimators



def initialize_model(estimation_method, model, transactions, products):

    model_info = oda_estimators[estimation_method]['models'][model]

    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )


    model = model_info['model_class'](products)

    if hasattr(model_info['estimator'], 'estimate_with_market_discovery'):
        result = model_info['estimator'].estimate_with_market_discovery(model, transactions)
    else:
        result = model_info['estimator'].estimate(model, transactions)

    return result


def EstimateErrorBound(estimation_method, model, products, in_sample_transactions, out_of_sample_transactions_prob):
    # model_results: initilized model
    result = initialize_model(estimation_method, model, in_sample_transactions, products)
    # result = model_info['estimator'].estimate(model, in_sample_transactions)
    assortments = result.GenerateAssormentData(out_of_sample_transactions_prob)
    empirical_dict = EmpiricalEstimator(in_sample_transactions).empirical_dict
    sse = {}
    for key, sample_transactions in assortments.items():

        sse.update({key: {'sse_m': result.sum_of_squared_error(sample_transactions),
                   'sse_e':result.sse_empirical(sample_transactions, empirical_dict),
                    'cross': result.cross_product_model_empirical(sample_transactions, empirical_dict)}})


    #return a dictionary with assorments as the key, and the sse as the value
    # print("EstimateErrorBound----------------")
    # print(sse)
    return sse




def AverageErrorBound(estimation_method, model, products, validating_transactions, validating_prob):
    K_train = 5
    # pool = Pool(mp.cpu_count() - 2)
    results = []
    for k in range(K_train):
        train_result = EstimateErrorBound(estimation_method, model, products, validating_transactions, validating_prob)
        # record the result
        results.append(train_result)

    avg_sse = {}

    #sum the errors
    # the average error is for each model with each assortments
    for error_dict in results:
        for key_assortment in error_dict.keys():
            if key_assortment in avg_sse:
                # for key_e in error_dict[key_assortment]:
                val_e = avg_sse[key_assortment]['sse_e']
                val_e += error_dict[key_assortment]['sse_e']
                val_m = avg_sse[key_assortment]['sse_m']
                val_m += error_dict[key_assortment]['sse_m']
                val_c = avg_sse[key_assortment]['cross']
                val_c +=  error_dict[key_assortment]['cross']
                avg_sse.update({key_assortment: {'sse_e': val_e, 'sse_m': val_m, 'cross': val_c}})
            else:
                avg_sse.update({key_assortment: {'sse_e': error_dict[key_assortment]['sse_e'],
                                                 'sse_m': error_dict[key_assortment]['sse_m'],
                                                 'cross': error_dict[key_assortment]['cross']}})

    # average the errors
    for key_assor in avg_sse:
        for key_e, value in avg_sse[key_assor].items():
            val =  value/K_train
            avg_sse[key_assor].update({key_e: val})
    # print("Average error bound")
    # print(avg_sse)
    return avg_sse



#calculate the alpha for particular s
def CalculateAlpha(estimation_method, model, products, avg_sse, in_sample_transactions):
    #separate the in sample transaction data into different assortments
    result = initialize_model(estimation_method, model, in_sample_transactions, products)
    assortments = result.GenerateAssormentData(in_sample_transactions)
    alpha_dict = {}
    for key_assor, transactions in assortments.items():
        # n_sample = len(transactions)
        # n_product = len(transactions[0].offered_products)
        if key_assor in avg_sse:
            sse_e = avg_sse[key_assor]['sse_e']
            sse_m = avg_sse[key_assor]['sse_m']
            cross = avg_sse[key_assor]['cross']
            # print(sse_e, sse_m, cross)
            if float(sse_m - 2 * cross + sse_e) != 0:
                a = (sse_e-cross)/float(sse_m - 2 * cross + sse_e)
            else:
                a = 0
            if a < 0:
                a = 0
            elif a > 1:
                a = 1
            alpha_dict.update({key_assor: a})
        else:
            print("Loss assortment, method:%s, model: %s, assort: %s" % (estimation_method, model, key_assor))
            print(avg_sse)
    # print("Alpha dict---------------")
    # print(alpha_dict)
    return alpha_dict


def get_alpha(estimation_method, model, products, validating_transactions, validating_prob):
    avg_sse = AverageErrorBound(estimation_method, model, products, validating_transactions, validating_prob)
    alpha_dict = CalculateAlpha(estimation_method, model, products, avg_sse, validating_transactions)
    # print('alpha_dict--------------------')
    # print(alpha_dict)
    return alpha_dict

