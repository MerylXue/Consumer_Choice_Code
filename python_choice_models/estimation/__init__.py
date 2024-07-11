# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.profiler import Profiler


class Estimator(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')


class EstimatorLevel(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def estimate(self, model, transactions, level):
        raise NotImplementedError('Subclass responsibility')
