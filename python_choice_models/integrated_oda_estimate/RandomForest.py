## This code is from the paper:
## Ningyuan Chen, Guillermo Gallego, Zhuodong Tang "The Use of Binary Choice Forests to Model and Estimate Discrete Choices"
## https://arxiv.org/abs/1908.01109

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from python_choice_models.estimation import Estimator
from math import sqrt, log
import json
import time
from python_choice_models.transactions.base import Transaction
from python_choice_models.transactions.base import Transaction_Extend
from python_choice_models.utils import safe_log
# random.seed(10)

## the binary choice forest method in "The Use of Binary Choice Forests to Model and Estimate Discrete Choices"

def print_decision_rules(rf, output):

    for tree_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1 # no support for multi-output

        output.write('TREE: {}\n'.format(tree_idx))

        iterator = enumerate(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
        for node_idx, data in iterator:
            left, right, feature, th, value = data

            # left: index of left child (if any)
            # right: index of right child (if any)
            # feature: index of the feature to check
            # th: the threshold to compare against
            # value: values associated with classes

            # for classifier, value is 0 except the index of the class to return
            class_idx = np.argmax(value[0])

            if left == -1 and right == -1:
                output.write('{} LEAF: return class={}\n'.format(node_idx, class_idx))
            else:
                output.write('{} NODE: if feature[{}] < {} then next={} else next={}\n'.format(node_idx, feature, th, left, right))

class RandomForestEstimator(Estimator):
    def __init__(self, n):
        self.n = n
        self.classifier = RandomForestClassifier(n_estimators = 1000, max_features = 'auto',
                                           random_state = 10, min_samples_split = 50)


        self.missing_features = np.ones(self.n + 1)

    def can_estimate(self):
        raise NotImplementedError('Subclass responsibility')

    def estimate_rf(self, in_sample_transactions):
        choices = []
        offered_set = []
        for transaction in in_sample_transactions:
            offered_set.append(transaction.offered_products)
            choices.append(transaction.product)

        T = len(choices)
        for i in range(self.n + 1):
            if i in choices:
                self.missing_features[i] = 0

        offered_set_binary = np.zeros((T, self.n + 1), int)
        for t in range(T):
            for l in range(len(offered_set[t])):
                offered_set_binary[t][offered_set[t][l]] = 1
        # print(offered_set_binary[240])
        self.classifier.fit(offered_set_binary, choices)
        return self.classifier

    def error_rf(self, in_sample_transactions, out_of_sample_transactions):
        self.estimate_rf(in_sample_transactions)
        rmse = 0.0
        ae = 0.0
        mrmse = 0.0
        mae = 0.0
        amount_terms = 0
        amount_ae = 0

        for transaction in out_of_sample_transactions:

            probabilities = self.probability_of(transaction)
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                index = transaction.offered_products.index(product)
                probability_1 = probabilities[product]
                probability_2 = transaction.prob[index]
                # print(probability_1, probability_2)
                # probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability_1 - probability_2) ** 2)
                # print(rmse)
                ae += abs(probability_1 - probability_2)
                # index += 1
                if num_product > 1:
                    mrmse += ((probability_1 - probability_2) ** 2) * num_product / (num_product - 1)
                    # mae += abs(probability_1 - probability_2) * num_product / (num_product - 1)
                    mae += abs(probability_1 - probability_2) * 1/sqrt(num_product)
                amount_terms += 1
            amount_ae += 1

        # print(rmse, ae, amount_terms)
        # print(sqrt(rmse / float(amount_terms)))

        return sqrt(rmse / float(amount_terms)), ae/ float(amount_ae), sqrt(mrmse / float(amount_ae)), mae / float(amount_ae)

    def probability_of(self, transaction):
        offered_set_binary = np.zeros(self.n + 1, int)

        for product in transaction.offered_products:
            offered_set_binary[product] = 1

        rf_test = self.classifier.predict_proba([offered_set_binary])
        rf_test = rf_test.flatten()

        # print(self.missing_features)
        for l in range(self.n + 1):
            num_insert = 0
            if self.missing_features[l] != 0:
                a_l = rf_test.tolist()
                a_l.insert(l + num_insert, 0)
                rf_test = np.asarray(a_l)

        predict_prob = offered_set_binary * rf_test
        if np.sum(predict_prob) != 0:
            probabilities = predict_prob / np.sum(predict_prob)
        else:
            probabilities = predict_prob

        return probabilities

    def log_probability_of(self, transaction):
        probabilities = self.probability_of(transaction)

        return safe_log(probabilities[transaction.product])

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


    def AIC_rf(self, transactions):
        # number of parameters
        # len of in_sample_transactions *
        k = 1000
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions)
        # return - 2 * l / amount_samples + 2 * k / amount_samples
        return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))

    def BIC_rf(self, transactions):
        k = 1000
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions)
        return -2 * l + (k * log(amount_samples))

    def hard_chi_squared_score_for(self, transactions, products):
        expected_purchases = [0.0 for _ in products]
        observed_purchases = [0.0 for _ in products]

        for transaction in transactions:
            observed_purchases[transaction.product] += 1.0
            probabilities = self.probability_of(transaction)
            for product in transaction.offered_products:
                expected_purchases[product] += probabilities[transaction.product]

        score = 0.0
        for p in products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(products))

def run_with_rf(file_name, N_prod):
    input_file = open(file_name, 'r')
    data = json.loads(input_file.read())
    input_file.close()
    products = list(range(data['amount_products']))


    ##In sample: data for training
    # Data format: dict{"amount_products":X, "transactions":{{"in_sample"...}{"out_of_sample"..,}}
    # Each data piece {"products":xx."offered_products":[id of products]}
    in_sample_transactions = Transaction.from_json(data['transactions']['in_sample'])
    in_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['in_sample_prob'])
    #Out of sample: choice probability of test data
    out_of_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['out_of_sample_prob'])
    all_sample_transactions_prob = Transaction_Extend.from_json_prob(data['transactions']['all_sample_prob'])
    # print(in_sample_transactions)
    model = RandomForestEstimator(N_prod)
    t1 = time.time()
    model.estimate_rf(in_sample_transactions)
    t2 = time.time()
    error_dict = {}
    rmse_in, ae_in, mrmse_in, mae_in = model.error_rf(in_sample_transactions, in_sample_transactions_prob)
    error_dict.update({"rmse_in": rmse_in, "ae_in": ae_in, "mrmse_in": mrmse_in, "mae_in": mae_in})
    rmse_out, ae_out, mrmse_out, mae_out = model.error_rf(in_sample_transactions, out_of_sample_transactions_prob)
    error_dict.update({"rmse_out": rmse_out, "ae_out": ae_out, "mrmse_out": mrmse_out, "mae_out": mae_out})
    rmse_all, ae_all, mrmse_all, mae_all = model.error_rf(in_sample_transactions, all_sample_transactions_prob)
    error_dict.update({"rmse_all": rmse_all, "ae_all": ae_all, "mrmse_all": mrmse_all, "mae_all": mae_all})
    AIC = 0
    BIC = 0
    chi2 = model.hard_chi_squared_score_for(in_sample_transactions, products)
    error_dict.update({"AIC": AIC, "BIC": BIC, "chi2": chi2})
    error_dict.update({"time": t2 - t1})
    error_dict.update({"num_iter": 0})
    # avg_ae = model.ae_known_prob(out_of_sample_transactions_prob)
    return error_dict
