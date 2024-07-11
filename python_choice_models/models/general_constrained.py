# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.



# This code is developing a choice model
# requiring probabilities statisfying monotonicity and supermodularity

from numpy import ones

from python_choice_models.models import Model
from python_choice_models.utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_m, ZERO_LOWER_BOUND, \
    array_to_assort_list, Lattice
from python_choice_models.optimization.non_linear import Constraints

import numpy as np
import itertools

class GeneralConstrainedModel(Model):
    @classmethod
    def code(cls):
        return 'gc'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['probabilities'])

    @classmethod
    def simple_deterministic(cls, products):
        # generate equal probabilities
        lst = list(itertools.product([0, 1], repeat=len(products) - 1))
        func = lambda x, y: x * y
        probabilities = np.zeros((len(lst), len(products)))
        for k in range(len(lst)):
            probabilities[k] = list(map(func, generate_n_equal_numbers_that_sum_one(len(products)), [1] + list(lst[k])))
            probabilities[k] = probabilities[k] / np.sum(probabilities[k])

        return cls(products, probabilities)

    @classmethod
    def simple_random(cls, products):
        lst = list(itertools.product([0, 1], repeat=len(products) - 1))
        func = lambda x, y: x * y
        probabilities = np.zeros((len(lst), len(products)))
        for k in range(len(lst)):
            probabilities[k] = list(
                map(func, generate_n_random_numbers_that_sum_m(len(products), 1), [1] + list(lst[k])))
            probabilities[k] = probabilities[k] / np.sum(probabilities[k])

        return cls(products, probabilities)


    def __init__(self, products, probabilities):
        super(GeneralConstrainedModel, self).__init__(products)
        if len(probabilities) != 2 ** (len(products) - 1):
            info = (len(probabilities), len(products))
            raise Exception('Incorrect amount of probabilities (%s) for amount of products (%s).' % info)

        self.probabilities = probabilities
        lst = list(itertools.product([0, 1], repeat=len(self.products) - 1))
        self.empirical_dict = dict()

        self.empirical_prob = np.zeros((len(lst), len(products)))
        self.modified_prob = np.zeros((len(lst), len(products)))
        self.centroid_prob = np.zeros((len(lst), len(products)))
        self.true_prob = np.zeros((len(lst), len(products)))


    def probability_of(self, transaction):
        idx = self.assortment_index(transaction)
        return self.probabilities[idx][transaction.product]

    def True_prob(self, transactions):
        for transaction in transactions:
            index = 0
            assort_idx = self.assortment_index(transaction)
            for product in transaction.offered_products:
                self.true_prob[assort_idx][product] = transaction.prob[index]
                index += 1
        return self.true_prob

    def parameters_vector(self):
        return self.probabilities

    def update_parameters_from_vector(self, probabilities):
        self.probabilities = list(probabilities)

    def constraints(self):
        return GeneralConstrainedModelConstraints(self)

    def Prob_Satisfied_CCP_Constraints(self):
        satisfy = True
        n_products = len(self.products) - 1
        for B in self.index_dict:
            B_lst = array_to_assort_list(B, n_products)
            M = [i + 1 for i in range(n_products)]

            idx_B = self.index_dict[tuple(B)]
            for i in range(1, n_products + 1):
                if  self.probabilities[idx_B][i] > B[i - 1]:
                    satisfy = False
                    print("Violate x[%d, %d] <= B[%d]" % (idx_B, i, i - 1))
                    print(self.probabilities[idx_B])
                    break

            if abs(sum(self.probabilities[idx_B]) - 1) > 1e-6:
                satisfy = False
                print("choice prob not sum to 1: %d" % idx_B)
                print(self.probabilities[idx_B])
                break

            lattice = Lattice(B_lst,M)
            idx_lst = []
            for k in range(len(lattice)):
                # print(tuple([1 if i in lattice[k] else 0 for i in range(1, self.n_products + 1)]))
                idx_k = self.index_dict[
                    tuple([1 if i in lattice[k] else 0 for i in range(1, n_products+1)])]
                idx_lst.append((idx_k, k))

            for item in [0] + B_lst:
            # for item in B_lst:
                if sum([self.probabilities[a[0]][item] * ((-1) ** (len(lattice[a[1]]) - len(B_lst))) for a in
                        idx_lst]) < -1e-6:
                    satisfy = False
                    print("Violate high-order monotone")
                    print(B_lst, item, lattice,
                          [self.probabilities[a[0]][item] * (-1) ** (len(lattice[a[1]]) - len(B_lst)) for a in idx_lst])
                    break

        return satisfy

    def Prob_Satisfied_SUP_Constraints(self):
        satisfy = True
        n_products = len(self.products) - 1
        for B in self.index_dict:
            zero_index = [i for i in range(len(B)) if B[i] == 0]
            one_index = [i for i in range(len(B)) if B[i] == 1]


            idx_B = self.index_dict[tuple(B)]
            for i in range(1, n_products + 1):
                if self.probabilities[idx_B][i] > B[i - 1]:
                    satisfy = False
                    print("Violate x[%d, %d] <= B[%d]" % (idx_B, i, i - 1))
                    print(self.probabilities[idx_B])

            if abs(sum(self.probabilities[idx_B]) - 1) > 1e-6:
                satisfy = False
                print("choice prob not sum to 1: %d" % idx_B)
                print(self.probabilities[idx_B])

            for k in zero_index:
                B_add = list(B).copy()
                B_add[k] = 1
                idx_B_add = self.index_dict[tuple(B_add)]
                # monotonicity constraint
                # opt_m.addConstr(x[idx_B, 0] >= x[idx_B_add, 0])
                for i in one_index + [-1]:
                    if self.probabilities[idx_B][i+1] - self.probabilities[idx_B_add][i+1] < -1e-6:
                        satisfy = False
                        print("Violate monotonicity")
                        print(B, B_add, i+1,  self.probabilities[idx_B][i+1], self.probabilities[idx_B_add][i+1])

                for k2 in zero_index:
                    if k2 != k:
                        B_add2 = list(B).copy()
                        B_add2[k2] = 1
                        idx_B_add2 = self.index_dict[tuple(B_add2)]
                        B_add_both = list(B_add).copy()
                        B_add_both[k2] = 1
                        idx_B_add_both = self.index_dict[tuple(B_add_both)]
                        for i in one_index + [-1]:
                            if self.probabilities[idx_B][i+1] - self.probabilities[idx_B_add][i+1]   - self.probabilities[idx_B_add2][i+1] +  self.probabilities[idx_B_add_both][i+1] < -1e-6:
                                satisfy = False
                                print("Violate supermodularity")
                                print(B, B_add, B_add2, B_add_both, self.probabilities[idx_B][i+1] - self.probabilities[idx_B_add][i+1],  self.probabilities[idx_B_add2][i+1] - self.probabilities[idx_B_add_both][i+1]  )
        return satisfy


    def __repr__(self):
        return '<Products: %s ; Probabilities: %s >' % (self.products, self.probabilities)

class GeneralConstrainedModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector()))


