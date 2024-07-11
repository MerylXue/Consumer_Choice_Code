# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.



from python_choice_models.transactions.base import Transaction_Extend
from math import sqrt, log
import time
import json
from python_choice_models.utils import safe_log
from python_choice_models.transactions.base import Transaction
from python_choice_models.estimation import Estimator
##generate empirical estimation probability

class EmpiricalEstimator(Estimator):
    def __init__(self, transactions):
        self.empirical_dict = dict()
        for transaction in transactions:
            # print(in_sample_transaction)
            # print(tuple(transaction.offered_products))
            if tuple(transaction.offered_products) in self.empirical_dict:
                key_a = tuple(transaction.offered_products)
                if transaction.product in self.empirical_dict[key_a]:
                    key_b = transaction.product
                    val = self.empirical_dict[key_a][key_b] + 1
                else:
                    key_b = transaction.product
                    val = 1
                self.empirical_dict[key_a].update({key_b: val})

                if 'count_assor' in self.empirical_dict[key_a]:
                    val2 = self.empirical_dict[key_a]['count_assor'] + 1
                    self.empirical_dict[key_a].update({'count_assor': val2})
                else:
                    self.empirical_dict[key_a].update({'count_assor': 1})
            else:
                key_a = tuple(transaction.offered_products)
                key_b = transaction.product
                val = 1
                self.empirical_dict.update({key_a: {key_b: val}})
                self.empirical_dict[key_a].update({'count_assor': 1})


        for key_1, value in self.empirical_dict.items():
            count_assortment = int(self.empirical_dict[key_1]['count_assor'])
            for key_2, value2 in value.items():
                if key_2 != 'count_assor':
                    val = float(value2 / count_assortment)
                    value.update({key_2: val})




    def error_emprical(self, transactions):
        rmse = 0.0
        mrmse = 0.0
        ae = 0.0
        mae = 0.0
        amount_terms_rmse = 0
        amount_terms_ae = 0
        # print(transactions)
        for transaction in transactions:
            index = 0
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_e = 0.0
                if tuple(transaction.offered_products) in self.empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if product in self.empirical_dict[key_a]:
                        key_b = product
                        probability_e = self.empirical_dict[key_a][key_b]
                # else:
                #     probability_e = 0.0
                # print(transaction)
                probability_1 = transaction.prob[index]
                index += 1

                rmse += ((probability_1 - probability_e) ** 2)
                if num_product > 1:
                    mrmse += ((probability_1 - probability_e) ** 2) * num_product / (num_product - 1)
                else:
                    mrmse += ((probability_1 - probability_e) ** 2)

                ae += abs(probability_1 - probability_e)

                if num_product > 1:
                    mae += abs(probability_1 - probability_e) * 1/sqrt(num_product)
                else:
                    mae += abs(probability_1 - probability_e)
                amount_terms_rmse += 1
            amount_terms_ae += 1

        return sqrt(rmse / float(amount_terms_rmse)), ae/float(amount_terms_ae), \
               sqrt(mrmse/float(amount_terms_ae)), mae/float(amount_terms_ae)


    def probability_of(self, transaction):
        probability_e = 0.0
        if tuple(transaction.offered_products) in self.empirical_dict:
            key_a = tuple(transaction.offered_products)
            if transaction.product in self.empirical_dict[key_a]:
                key_b = transaction.product
                probability_e = self.empirical_dict[key_a][key_b]
        return probability_e

    def log_probability_of(self, transaction):
        probability_e = self.probability_of(transaction)
        return safe_log(probability_e)

    def log_likelihood_for(self, transactions):
        result = 0
        cache = {}
        for transaction in transactions:
            cache_code = (transaction.product, tuple(transaction.offered_products))
            if cache_code in cache:
                log_probability = cache[cache_code]
            else:
                log_probability = self.log_probability_of(transaction)
                cache[cache_code] = log_probability
            result += log_probability
        return result

    def AIC_empirical(self, transactions):
        # number of parameters
        # len of in_sample_transactions *
        k = sum([len(key) for key in self.empirical_dict])
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions)
        # return - 2 * l / amount_samples + 2 * k / amount_samples
        return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))

    def BIC_empirical(self, transactions):
        k = sum([len(key) for key in self.empirical_dict])
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions)
        return -2 * l + (k * log(amount_samples))

    def hard_chi_squared_score_for(self, transactions, products):
        expected_purchases = [0.0 for _ in products]
        observed_purchases = [0.0 for _ in products]

        for transaction in transactions:
            observed_purchases[transaction.product] += 1.0
            for product in transaction.offered_products:
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(products))

def run_with_empirical(file_name):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))

    # Ground_model = str(data['Ground'])
    # ini_param = data['ini_param']

    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])

    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])
    # print(in_sample_transactions)
    # print(in_sample_transactions_prob)
    t1 = time.time()
    model = EmpiricalEstimator(in_sample_transactions)
    t2 = time.time()
    error_dict = {}
    rmse_in, ae_in, mrmse_in, mae_in = model.error_emprical(in_sample_transactions_prob)
    error_dict.update({"rmse_in":rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in":mae_in})
    # rmse_out, ae_out, mrmse_out, mae_out = model.error_emprical(out_of_sample_transactions_prob)
    error_dict.update({"rmse_out": 0, "ae_out": 0, "mrmse_out": 0, "mae_out":0})
    # rmse_all, ae_all, mrmse_all, mae_all = model.error_emprical(all_sample_transactions_prob)
    error_dict.update({"rmse_all": 0, "ae_all":0, "mrmse_all": 0, "mae_all": 0})
    AIC = 0
    BIC = 0
    chi2 = model.hard_chi_squared_score_for(in_sample_transactions, products)
    # print("Empirical AIC: %f, BIC: %f, chi2: %f" % (AIC, BIC, chi2))
    error_dict.update({"AIC": AIC, "BIC": BIC, "chi2": chi2})
    error_dict.update({"time": time})
    return error_dict

