# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

from math import log, sqrt
from python_choice_models.transactions.base import Transaction, Transaction_Extend
from python_choice_models.utils import safe_log
import json
import itertools
import numpy as np
from copy import deepcopy

class Model(object):
    """
        Represents a mathematical model for Discrete Choice Consumer Decision.
    """
    def __init__(self, products):
        if products != list(range(len(products))):
            raise Exception('Products should be entered as an ordered consecutive list.')
        self.products = products
        self.index_dict = {}

        self.in_sample_assort = {}
        self.out_of_sample_assort = {}

        lst = list(itertools.product([0, 1], repeat=len(self.products) - 1))
        idx = 0
        # list of binary representation of assortments
        self.assort_set = lst

        for l in lst:
            self.index_dict.update({l: idx})
            idx += 1

    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def from_data(cls, data):
        for klass in cls.__subclasses__():
            if data['code'] == klass.code():
                return klass.from_data(data)
        raise Exception('No model can be created from data %s')

    @classmethod
    def simple_deterministic(cls, *args, **kwargs):
        """
            Must return a default model with simple pdf parameters to use as an initial solution for estimators.
        """
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def simple_random(cls, *args, **kwargs):
        """
            Must return a default model with random pdf parameters to use as a ground model.
        """
        raise NotImplementedError('Subclass responsibility')

    def probability_of(self, transaction):
        """
            Must return the probability of a transaction.
        """
        raise NotImplementedError('Subclass responsibility')

    def log_probability_of(self, transaction):
        return safe_log(self.probability_of(transaction))

    def probability_distribution_over(self, offered_products):
        distribution = []
        for product in range(len(self.products)):
            transaction = Transaction(product, offered_products)
            distribution.append(self.probability_of(transaction))
        return distribution

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
        return result / len(transactions)

    def infinite_in_sample_log_likelihood(self, ground_model):
        result = 0
        for t in Transaction.all_for(self):
            result += (ground_model.probability_of(t) * self.log_probability_of(t))
        return result

    def soft_rmse_for(self, ground_model):
        rmse = 0.0
        amount_terms = 0.0
        for t in Transaction.all_for(self):
            rmse += ((self.probability_of(t) - ground_model.probability_of(t)) ** 2)
            amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def rmse_for(self, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability = self.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability - float(product == transaction.product)) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def rmse_known_ground(self, ground_model, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability_1 - probability_2) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def hit_rate_for(self, transactions):
        hit_rate = 0
        for transaction in transactions:
            probabilities = []
            for product in transaction.offered_products:
                probabilities.append((product, self.probability_of(Transaction(product, transaction.offered_products))))
            most_probable = max(probabilities, key=lambda p: p[1])[0]
            hit_rate += int(most_probable == transaction.product)
        return float(hit_rate) / float(len(transactions))

    def soft_chi_squared_score_for(self, ground_model, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            for product in transaction.offered_products:
                observed_purchases[product] += ground_model.probability_of(Transaction(product, transaction.offered_products))
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def hard_chi_squared_score_for(self, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            observed_purchases[transaction.product] += 1.0
            for product in transaction.offered_products:
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def aic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        if amount_samples - k - 1 > 0:
            return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))
        else:
            return 2 * (k - l )

    def bic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        return -2 * l + (k * log(amount_samples))

    def amount_of_parameters(self):
        return len(self.parameters_vector())

    def save(self, file_name):
        with open(file_name, 'w+') as f:
            f.write(json.dumps(self.data(), indent=1))

    #####################################2022.08###################################################
    def sum_of_squared_error(self, transactions):
        sse = 0.0
        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = transaction.prob[index]
                sse += ((probability_1 - probability_2) ** 2)
                index += 1
        return sse

    def sum_of_squared_error_smooth(self, transactions, empirical_dict, alpha):
        sse = 0.0
        msse = 0.0
        for transaction in transactions:
            index = 0
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if n_sample > 0:
                    probability_1 = alpha * probability_m + (1 - alpha) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                sse += ((probability_1 - probability_2) ** 2)
                if num_product > 1:
                    msse += ((probability_1 - probability_2) ** 2) * num_product / (num_product - 1)
                else:
                    msse = sse
                index += 1
        return sse, msse



    def sum_of_absolute_error_smooth(self, transactions, empirical_dict, alpha):
        sae = 0.0
        msae = 0.0
        for transaction in transactions:
            index = 0
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if n_sample > 0:
                    probability_1 = alpha * probability_m + (1 - alpha) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                sae += abs(probability_1 - probability_2)

                if num_product > 1:
                    msae += abs(probability_1 - probability_2)  * num_product / (num_product - 1)
                else:
                    msae = sae
                index += 1

        return sae, msae

    def sse_empirical(self, transactions, empirical_dict):
        sse = 0.0
        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                probability_1 = 0.0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)


                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_1 = empirical_dict[key_a][key_b]
                probability_2 = transaction.prob[index]
                sse += ((probability_1 - probability_2) ** 2)
                index += 1
        return sse

    def cross_product_model_empirical(self, transactions, empirical_dict):
        sse = 0.0
        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                probability_e = 0.0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]
                probability_2 = transaction.prob[index]
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                sse += (probability_e - probability_2 ) * (probability_m - probability_2)
                index += 1
        return sse
    def rmse_known_prob(self, transactions):
        rmse = 0.0
        amount_terms = 0

        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = transaction.prob[index]
                rmse += ((probability_1 - probability_2) ** 2)
                index += 1
                amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    # def rmse_known_prob_with_modified_transactions(self, transactions, modified_probabilities, out_of_sample_assort):
    #     rmse = 0.0
    #     amount_terms = 0
    #
    #     for transaction in transactions:
    #         index = 0
    #         assort_index = self.assortment_index(transaction)
    #         if assort_index in out_of_sample_assort.values():
    #             for product in transaction.offered_products:
    #                 probability_1 = modified_probabilities[assort_index][product]
    #                 probability_2 = transaction.prob[index]
    #                 rmse += ((probability_1 - probability_2) ** 2)
    #                 index += 1
    #                 amount_terms += 1
    #         else:
    #             for product in transaction.offered_products:
    #                 probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
    #                 probability_2 = transaction.prob[index]
    #                 rmse += ((probability_1 - probability_2) ** 2)
    #                 index += 1
    #                 amount_terms += 1
    #
    #     return sqrt(rmse / float(amount_terms))

    def ae_known_prob(self, transactions):
        ae = 0.0
        amount_terms = 0
        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = transaction.prob[index]
                # probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                ae += abs(probability_1 - probability_2)
                index += 1
            amount_terms += 1
        return ae / float(amount_terms)

    # def ae_known_prob_with_modified_transactions(self, transactions, modified_probabilities, out_of_sample_assort):
    #     ae = 0.0
    #     amount_terms = 0
    #
    #     for transaction in transactions:
    #         index = 0
    #         assort_index = self.assortment_index(transaction)
    #         if assort_index in out_of_sample_assort.values():
    #             for product in transaction.offered_products:
    #                 probability_1 = modified_probabilities[assort_index][product]
    #                 probability_2 = transaction.prob[index]
    #                 ae += abs(probability_1 - probability_2)
    #                 index += 1
    #
    #         else:
    #             for product in transaction.offered_products:
    #                 probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
    #                 probability_2 = transaction.prob[index]
    #                 ae += ((probability_1 - probability_2) ** 2)
    #                 index += 1
    #         amount_terms += 1
    #
    #     return ae / float(amount_terms)



    def mrmse_known_prob(self, transactions):
        rmse = 0.0
        amount_terms = 0

        for transaction in transactions:
            index = 0
            num_products = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = transaction.prob[index]
                # probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                if (num_products - 1) > 0:
                    rmse += ((probability_1 - probability_2) ** 2) * num_products/(num_products - 1)
                else:
                    rmse += ((probability_1 - probability_2) ** 2)
                index += 1
            amount_terms += 1
        return sqrt(rmse / float(amount_terms))


    # def mrmse_known_prob_with_modified_transactions(self, transactions, modified_probabilities, out_of_sample_assort):
    #     rmse = 0.0
    #     amount_terms = 0
    #
    #     for transaction in transactions:
    #         index = 0
    #         num_products = len(transaction.offered_products)
    #         assort_index = self.assortment_index(transaction)
    #         if assort_index in out_of_sample_assort.values():
    #             for product in transaction.offered_products:
    #                 probability_1 = modified_probabilities[assort_index][product]
    #                 probability_2 = transaction.prob[index]
    #                 if (num_products - 1) > 0:
    #                     rmse += ((probability_1 - probability_2) ** 2) * num_products/(num_products - 1)
    #                 else:
    #                     rmse += ((probability_1 - probability_2) ** 2)
    #                 index += 1
    #         else:
    #             for product in transaction.offered_products:
    #                 probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
    #                 probability_2 = transaction.prob[index]
    #                 if (num_products - 1) > 0:
    #                     rmse += ((probability_1 - probability_2) ** 2) * num_products/(num_products - 1)
    #                 else:
    #                     rmse += ((probability_1 - probability_2) ** 2)
    #                 index += 1
    #         amount_terms += 1
    #     return sqrt(rmse / float(amount_terms))

    def mae_known_prob(self, transactions):
        ae = 0.0
        amount_terms = 0

        for transaction in transactions:
            index = 0
            num_products = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = transaction.prob[index]
                if num_products > 1:
                    ae += abs(probability_1 - probability_2) * 1/sqrt(num_products)
                else:
                    ae += abs(probability_1 - probability_2)
                index += 1
            amount_terms += 1
        return ae / float(amount_terms)


    # def mae_known_prob_with_modified_transactions(self, transactions, modified_probabilities, out_of_sample_assort):
    #     ae = 0.0
    #     amount_terms = 0
    #
    #     for transaction in transactions:
    #         index = 0
    #         num_products = len(transaction.offered_products)
    #         assort_index = self.assortment_index(transaction)
    #         if assort_index in out_of_sample_assort.values():
    #             for product in transaction.offered_products:
    #                 probability_1 = modified_probabilities[assort_index][product]
    #                 probability_2 = transaction.prob[index]
    #                 if num_products > 1:
    #                     ae += abs(probability_1 - probability_2) * 1 / sqrt(num_products)
    #                 else:
    #                     ae += abs(probability_1 - probability_2)
    #                 index += 1
    #         else:
    #             for product in transaction.offered_products:
    #                 probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
    #                 probability_2 = transaction.prob[index]
    #                 if num_products > 1:
    #                     ae += abs(probability_1 - probability_2) * 1 / sqrt(num_products)
    #                 else:
    #                     ae += abs(probability_1 - probability_2)
    #                 index += 1
    #         amount_terms += 1
    #     return ae / float(amount_terms)


    def ae_for(self, transactions):
        ae = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability = self.probability_of(Transaction(product, transaction.offered_products))
                ae += abs(probability - float(product == transaction.product))
            amount_terms += 1
        return ae / float(amount_terms)

    def mrmse_known_ground(self, ground_model, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                if num_product > 1:
                    rmse += ((probability_1 - probability_2) ** 2) * num_product/(num_product - 1)
                else:
                    rmse += ((probability_1 - probability_2) ** 2)
            amount_terms += 1
        return sqrt(rmse / float(amount_terms))


    def GenerateAssormentData(self, transactions):
        # depart each assortment
        # output: a dict with key as offered_products, value as a list of the data
        # associated with the assortment
        assortment = {}
        for transaction in transactions:
            if tuple(transaction.offered_products) not in assortment:
                lst = []
                lst.append(transaction)
                assortment.update({tuple(transaction.offered_products): lst})
            else:
                lst = assortment[tuple(transaction.offered_products)]
                lst.append(transaction)
                assortment.update({tuple(transaction.offered_products): lst})
        return assortment

    def assortment_index(self, transaction):

        binary_transaction = [0 for l in range(len(self.products) - 1)]
        for product in transaction.offered_products:
            if product != 0:
                binary_transaction[product - 1] = 1
        return self.index_dict[tuple(binary_transaction)]


    def sample_assortments(self, transactions):
        idx_in = 0
        tmp_in_sample_assort = []
        for transaction in transactions:
            idx = self.assortment_index(transaction)
            tmp_in_sample_assort.append(idx)

        tmp_in_sample_assort = list(set(tmp_in_sample_assort))
        for item in tmp_in_sample_assort:
            self.in_sample_assort.update({item: idx_in})
            idx_in += 1

        idx_out = 0
        for key, value in self.index_dict.items():
            if value not in self.in_sample_assort:
                self.out_of_sample_assort.update({value: idx_out})
                idx_out += 1
    def GenerateEmpricalDict(self, transactions):
        empirical_dict = {}
        for transaction in transactions:
            # print(in_sample_transaction)
            # print(tuple(transaction.offered_products))
            if tuple(transaction.offered_products) in empirical_dict:
                key_a = tuple(transaction.offered_products)
                if transaction.product in empirical_dict[key_a]:
                    key_b = transaction.product
                    val = empirical_dict[key_a][key_b] + 1
                else:
                    key_b = transaction.product
                    val = 1
                empirical_dict[key_a].update({key_b: val})

                if 'count_assor' in empirical_dict[key_a]:
                    val2 = empirical_dict[key_a]['count_assor'] + 1
                    empirical_dict[key_a].update({'count_assor': val2})
                else:
                    empirical_dict[key_a].update({'count_assor': 1})
            else:
                key_a = tuple(transaction.offered_products)
                key_b = transaction.product
                val = 1
                empirical_dict.update({key_a: {key_b: val}})
                empirical_dict[key_a].update({'count_assor': 1})


        for key_1, value in empirical_dict.items():
            count_assortment = int(empirical_dict[key_1]['count_assor'])
            for key_2, value2 in value.items():
                if key_2 != 'count_assor':
                    val = float(value2 / count_assortment)
                    value.update({key_2: val})
        return empirical_dict

    def GenerateOutofSampleTransactions(self, in_sample_transactions, empirical_dict):
        n_product = len(self.products)

        out_of_sample_transactions = deepcopy(in_sample_transactions)
        out_of_sample_transactions_prob = []
        for transaction in out_of_sample_transactions:
            offered_set = tuple(transaction.offered_products)
            choice_prob = np.zeros(n_product)
            offered_product_prob = np.zeros(len(transaction.offered_products))
            idx = 0
            for product in transaction.offered_products:
                if offered_set in empirical_dict:
                    if product in empirical_dict[offered_set]:
                        choice_prob[product] = empirical_dict[offered_set][product]
                        offered_product_prob[idx] = empirical_dict[offered_set][product]
                idx += 1


            choice = np.random.choice(n_product, 1, p=choice_prob)
            transaction.product = choice[0]
            transaction_prob = Transaction_Extend(offered_product_prob, offered_set)
            out_of_sample_transactions_prob.append(transaction_prob)

        return out_of_sample_transactions, out_of_sample_transactions_prob

    def kernel_smooth_RMSE_known_prob(self, transactions, empirical_dict, alpha):
        rmse = 0.0
        amount_terms = 0

        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if probability_e > 0:
                    probability_1 = (alpha ** (n_sample / size)) * probability_m + (
                                1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                # if (probability_e > 0):
                # print(probability_e, probability_m)
                # print(probability_1, probability_2)
                rmse += ((probability_1 - probability_2) ** 2)
                index += 1
                amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    def kernel_smooth_MRMSE_known_prob(self, transactions, empirical_dict, alpha):
        rmse = 0.0
        amount_terms = 0

        for transaction in transactions:
            index = 0
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if probability_e > 0:
                    probability_1 = (alpha ** (n_sample / size)) * probability_m + (
                                1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                # if (probability_e > 0):
                # print(probability_e, probability_m)
                # print(probability_1, probability_2)
                if num_product > 1:
                    rmse += ((probability_1 - probability_2) ** 2) * num_product/(num_product - 1)
                else:
                    rmse += ((probability_1 - probability_2) ** 2)
                index += 1
            amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    def kernel_smooth_AE_known_prob(self, transactions, empirical_dict, alpha):
        ae = 0.0
        amount_terms = 0
        for transaction in transactions:
            index = 0
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if probability_e > 0:
                    probability_1 = (alpha ** (n_sample / size)) * probability_m + (
                                1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                # if (probability_e > 0):
                # print(probability_e, probability_m)
                # print(probability_1, probability_2)
                ae += abs(probability_1 - probability_2)
                index += 1
            amount_terms += 1

        return ae / float(amount_terms)

    # return the modified absolute error of the ODA result (see the paper for definition)
    # the coefficient of empirical probability alpha is the same for all assortments
    def kernel_smooth_MAE_known_prob(self, transactions, empirical_dict, alpha):
        ae = 0.0
        amount_terms = 0
        for transaction in transactions:
            index = 0
            num_product = len(transaction.offered_products)
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if probability_e > 0:
                    probability_1 = (alpha ** (n_sample / size)) * probability_m + (
                                1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability_1 = probability_m

                probability_2 = transaction.prob[index]
                # if (probability_e > 0):
                # print(probability_e, probability_m)
                # print(probability_1, probability_2)
                if num_product > 1:
                    # ae += abs(probability_1 - probability_2) * num_product/(num_product - 1)
                    ae += abs(probability_1 - probability_2) * 1/sqrt(num_product)
                else:
                    ae += abs(probability_1 - probability_2)
                index += 1
            amount_terms += 1

        return ae / float(amount_terms)

    # return the  root mean squared error of the ODA result
    # the coefficient of empirical probability alpha is defined by assortment, \alpha(S)
    def kernel_smooth_RMSE_assortment_dict(self, transactions, empirical_dict, alpha_dict):
        rmse = 0.0
        amount_terms = 0

        assort_transactions = self.GenerateAssormentData(transactions)

        for key_assortment, transactions in assort_transactions.items():
            for transaction in transactions:
                index = 0
                for product in transaction.offered_products:
                    probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                    probability_e = 0.0
                    n_sample = 0
                    if tuple(transaction.offered_products) in empirical_dict:
                        key_a = tuple(transaction.offered_products)
                        if 'count_assor' in empirical_dict[key_a]:
                            n_sample = empirical_dict[key_a]['count_assor']
                        else:
                            print("dict formation error!")

                        if product in empirical_dict[key_a]:
                            key_b = product
                            probability_e = empirical_dict[key_a][key_b]

                    if probability_e > 0:
                        if key_assortment in alpha_dict:
                            probability_1 = alpha_dict[key_assortment] * probability_m \
                                            + (1 - alpha_dict[key_assortment]) * probability_e
                        else:
                            print("Key assortment does not in alpha_dict")
                            probability_1 = probability_m

                    else:
                        probability_1 = probability_m

                    probability_2 = transaction.prob[index]
                    # if (probability_e > 0):
                    # print(probability_e, probability_m)
                    # print(probability_1, probability_2)
                    rmse += (probability_1 - probability_2) ** 2
                    index += 1
                    amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    # return the modified root mean squared error of the ODA result (see the paper for definition)
    # the coefficient of empirical probability alpha is defined by assortment, \alpha(S)
    def kernel_smooth_MRMSE_assortment_dict(self, transactions, empirical_dict, alpha_dict):
        rmse = 0.0
        amount_terms = 0

        assort_transactions = self.GenerateAssormentData(transactions)

        for key_assortment, transactions in assort_transactions.items():
            for transaction in transactions:
                index = 0
                num_product = len(transaction.offered_products)
                for product in transaction.offered_products:
                    probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                    probability_e = 0.0
                    n_sample = 0
                    if tuple(transaction.offered_products) in empirical_dict:
                        key_a = tuple(transaction.offered_products)
                        if 'count_assor' in empirical_dict[key_a]:
                            n_sample = empirical_dict[key_a]['count_assor']
                        else:
                            print("dict formation error!")

                        if product in empirical_dict[key_a]:
                            key_b = product
                            probability_e = empirical_dict[key_a][key_b]

                    if probability_e > 0:
                        if key_assortment in alpha_dict:
                            probability_1 = alpha_dict[key_assortment] * probability_m \
                                            + (1 - alpha_dict[key_assortment]) * probability_e
                        else:
                            print("Key assortment does not in alpha_dict")
                            probability_1 = probability_m

                    else:
                        probability_1 = probability_m

                    probability_2 = transaction.prob[index]
                    # if (probability_e > 0):
                    # print(probability_e, probability_m)
                    # print(probability_1, probability_2)
                    if num_product > 1:
                        rmse += ((probability_1 - probability_2) ** 2) * num_product/(num_product - 1)
                    else:
                        rmse += ((probability_1 - probability_2) ** 2)
                    index += 1
                amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    # return the  absolute error of the ODA result
    # the coefficient of empirical probability alpha is defined by assortment, \alpha(S)
    def kernel_smooth_AE_assortment_dict(self, transactions, empirical_dict, alpha_dict):
        ae = 0.0
        amount_terms = 0

        assort_transactions = self.GenerateAssormentData(transactions)

        for key_assortment, transactions in assort_transactions.items():
            for transaction in transactions:
                index = 0
                for product in transaction.offered_products:
                    probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                    probability_e = 0.0
                    n_sample = 0
                    if tuple(transaction.offered_products) in empirical_dict:
                        key_a = tuple(transaction.offered_products)
                        if 'count_assor' in empirical_dict[key_a]:
                            n_sample = empirical_dict[key_a]['count_assor']
                        else:
                            print("dict formation error!")

                        if product in empirical_dict[key_a]:
                            key_b = product
                            probability_e = empirical_dict[key_a][key_b]

                    if probability_e > 0:
                        if key_assortment in alpha_dict:
                            probability_1 = alpha_dict[key_assortment] * probability_m \
                                            + (1 - alpha_dict[key_assortment]) * probability_e
                        else:
                            print("Key assortment does not in alpha_dict")
                            probability_1 = probability_m

                    else:
                        probability_1 = probability_m

                    probability_2 = transaction.prob[index]
                    # if (probability_e > 0):
                    # print(probability_e, probability_m)
                    # print(probability_1, probability_2)
                    ae += abs(probability_1 - probability_2)
                    index += 1
                amount_terms += 1
        return ae / float(amount_terms)

    # return the modified absolute error of the ODA result (see the paper for definition)
    # the coefficient of empirical probability alpha is defined by assortment, \alpha(S)
    def kernel_smooth_MAE_assortment_dict(self, transactions, empirical_dict, alpha_dict):
        ae = 0.0
        amount_terms = 0

        assort_transactions = self.GenerateAssormentData(transactions)

        for key_assortment, transactions in assort_transactions.items():
            for transaction in transactions:
                index = 0
                num_product = len(transaction.offered_products)
                for product in transaction.offered_products:
                    probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                    probability_e = 0.0
                    n_sample = 0
                    if tuple(transaction.offered_products) in empirical_dict:
                        key_a = tuple(transaction.offered_products)
                        if 'count_assor' in empirical_dict[key_a]:
                            n_sample = empirical_dict[key_a]['count_assor']
                        else:
                            print("dict formation error!")

                        if product in empirical_dict[key_a]:
                            key_b = product
                            probability_e = empirical_dict[key_a][key_b]

                    if n_sample > 0:
                        if key_assortment in alpha_dict:
                            probability_1 = alpha_dict[key_assortment] * probability_m \
                                            + (1 - alpha_dict[key_assortment]) * probability_e
                        else:
                            print("Key assortment does not in alpha_dict")
                            probability_1 = probability_m

                    else:
                        probability_1 = probability_m

                    probability_2 = transaction.prob[index]
                    # if (probability_e > 0):
                    # print(probability_e, probability_m)
                    # print(probability_1, probability_2)
                    if num_product > 1:
                        # ae += abs(probability_1 - probability_2) * num_product/(num_product - 1)
                        ae += abs(probability_1 - probability_2) * 1/sqrt(num_product)
                    else:
                        ae += abs(probability_1 - probability_2)
                    index += 1
                amount_terms += 1

        return ae / float(amount_terms)

    # return the root mean squared error of the ODA result
    # the coefficient of empirical probability alpha is the same for all assortment
    def kernel_smooth_RMSE_for(self, transactions, empirical_dict, alpha):
        rmse = 0.0
        amount_terms = 0

        for transaction in transactions:
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if n_sample > 0:
                    probability = (alpha ** (n_sample / size)) * probability_m + (
                                1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability = probability_m

                rmse += ((probability - float(product == transaction.product)) ** 2)
                amount_terms += 1

        return sqrt(rmse / float(amount_terms))

    #return the absolute error of the ODA result
    # the coefficient of empirical probability alpha is the same for all assortment
    def kernel_smooth_AE_for(self, transactions, empirical_dict, alpha):
        ae = 0.0
        amount_terms = 0

        for transaction in transactions:
            for product in transaction.offered_products:
                size = len(transaction.offered_products)
                probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                probability_e = 0.0
                n_sample = 0
                if tuple(transaction.offered_products) in empirical_dict:
                    key_a = tuple(transaction.offered_products)
                    if 'count_assor' in empirical_dict[key_a]:
                        n_sample = empirical_dict[key_a]['count_assor']
                    else:
                        print("dict formation error!")

                    if product in empirical_dict[key_a]:
                        key_b = product
                        probability_e = empirical_dict[key_a][key_b]

                if n_sample > 0:
                    probability = (alpha ** (n_sample / size)) * probability_m + (
                            1 - alpha ** (n_sample / size)) * probability_e

                else:
                    probability = probability_m

                ae += abs(probability - float(product == transaction.product))
            amount_terms += 1

        return ae / float(amount_terms)

    def log_likelihood_of_kernel(self, transactions, empirical_dict, alpha_dict):
        result = 0
        cache = {}
        if alpha_dict:
            assort_transactions = self.GenerateAssormentData(transactions)

            for key_assortment, transactions in assort_transactions.items():
                for transaction in transactions:
                    cache_code = (transaction.product, tuple(transaction.offered_products))
                    if cache_code in cache:
                        log_probability = cache[cache_code]
                    else:
                        probability_m = self.probability_of(transaction)
                        probability_e = 0.0
                        n_sample = 0
                        if tuple(transaction.offered_products) in empirical_dict:
                            key_a = tuple(transaction.offered_products)
                            if 'count_assor' in empirical_dict[key_a]:
                                n_sample = empirical_dict[key_a]['count_assor']
                            else:
                                print("dict formation error!")
                            if transaction.product in empirical_dict[key_a]:
                                key_b = transaction.product
                                probability_e = empirical_dict[key_a][key_b]

                        if n_sample > 0:
                            if key_assortment in alpha_dict:
                                probability_1 = alpha_dict[key_assortment] * probability_m \
                                                + (1 - alpha_dict[key_assortment]) * probability_e
                            else:
                                print("Key assortment does not in alpha_dict")
                                probability_1 = probability_m

                        else:
                            probability_1 = probability_m
                        log_probability = safe_log(probability_1)
                        cache[cache_code] = log_probability
                    result += log_probability
        else:
            for transaction in transactions:
                cache_code = (transaction.product, tuple(transaction.offered_products))
                if cache_code in cache:
                    log_probability = cache[cache_code]
                else:
                    log_probability = self.log_probability_of(transaction)
                    cache[cache_code] = log_probability
                result += log_probability

        return result

    def AIC_kernel(self, transactions, empirical_dict, alpha_dict):
        k = self.amount_of_parameters() + len(alpha_dict)
        amount_samples = len(transactions)
        l = self.log_likelihood_of_kernel(transactions, empirical_dict, alpha_dict)
        return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))

    def BIC_kernel(self, transactions, empirical_dict, alpha_dict):
        k = self.amount_of_parameters() + len(alpha_dict)
        amount_samples = len(transactions)
        l = self.log_likelihood_of_kernel(transactions, empirical_dict, alpha_dict)
        return -2 * l + (k * log(amount_samples))

    def chi_square_kenerl(self, transactions, empirical_dict, alpha_dict):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]
        if alpha_dict:
            assort_transactions = self.GenerateAssormentData(transactions)
            for key_assortment, transactions in assort_transactions.items():
                for transaction in transactions:
                    observed_purchases[transaction.product] += 1.0
                    index = 0
                    num_product = len(transaction.offered_products)
                    for product in transaction.offered_products:
                        probability_m = self.probability_of(Transaction(product, transaction.offered_products))
                        probability_e = 0.0
                        n_sample = 0
                        if tuple(transaction.offered_products) in empirical_dict:
                            key_a = tuple(transaction.offered_products)
                            if 'count_assor' in empirical_dict[key_a]:
                                n_sample = empirical_dict[key_a]['count_assor']
                            else:
                                print("dict formation error!")
                            if transaction.product in empirical_dict[key_a]:
                                key_b = transaction.product
                                probability_e = empirical_dict[key_a][key_b]

                        if n_sample > 0:
                            if key_assortment in alpha_dict:
                                probability_1 = alpha_dict[key_assortment] * probability_m \
                                                + (1 - alpha_dict[key_assortment]) * probability_e
                            else:
                                print("Key assortment does not in alpha_dict")
                                probability_1 = probability_m

                        else:
                            probability_1 = probability_m

                        expected_purchases[product] += probability_1
        else:
            for transaction in transactions:
                observed_purchases[transaction.product] += 1.0
                for product in transaction.offered_products:
                    expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))


    ##############################################################################################
    @classmethod
    def load(cls, file_name):
        with open(file_name, 'r') as f:
            model = cls.from_data(json.loads(f.read()))
        return model

    def parameters_vector(self):
        """
            Vector of parameters that define the model. For example lambdas and ros in Markov Chain.
        """
        return []

    def update_parameters_from_vector(self, parameters):
        """
            Updates current parameters from input parameters vector
        """
        pass

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')

    def data(self):
        raise NotImplementedError('Subclass responsibility')


from python_choice_models.models.exponomial import ExponomialModel
from python_choice_models.models.latent_class import LatentClassModel
from python_choice_models.models.markov_chain import MarkovChainModel
from python_choice_models.models.markov_chain_rank_2 import MarkovChainRank2Model
from python_choice_models.models.multinomial_logit import MultinomialLogitModel
from python_choice_models.models.nested_logit import NestedLogitModel
from python_choice_models.models.ranked_list import RankedListModel
from python_choice_models.models.random_choice import RandomChoiceModel
from python_choice_models.models.mixed_logit import MixedLogitModel
from python_choice_models.models.generalized_stochastic_preference import GeneralizedStochasticPreferenceModel
from python_choice_models.models.general_constrained import GeneralConstrainedModel