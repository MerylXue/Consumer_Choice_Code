# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.estimation import EstimatorLevel, Estimator
from python_choice_models.settings import Settings
from python_choice_models.utils import time_for_optimization
import time


class ExpectationMaximizationEstimator(Estimator):
    def estimate(self, model, transactions):
        self.profiler().reset_convergence_criteria()
        self.profiler().update_time()
        model = self.custom_initial_solution(model, transactions)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())

        start_time = time.time()
        while True:
            self.profiler().start_iteration()
            model = self.one_step(model, transactions)
            likelihood = model.log_likelihood_for(transactions)
            # mrmse = model.rmse_for(transactions)
            self.profiler().stop_iteration(likelihood)
            # self.profiler().stop_iteration(mrmse)

            if self.profiler().should_stop() or (time.time() - start_time) > cpu_time:
                break

        return model

    def one_step(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def custom_initial_solution(self, model, transactions):
        return model


class ExpectationMaximizationEstimatorLevel(EstimatorLevel):
    def estimate(self, model, transactions, level):
        self.profiler().reset_convergence_criteria()
        self.profiler().update_time()
        model = self.custom_initial_solution(model, transactions)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())

        start_time = time.time()
        while True:
            self.profiler().start_iteration()
            model = self.one_step(model, transactions, level)
            likelihood = model.log_likelihood_for(transactions)
            # mrmse = model.rmse_for(transactions)
            self.profiler().stop_iteration(likelihood)
            # self.profiler().stop_iteration(mrmse)

            if self.profiler().should_stop() or (time.time() - start_time) > cpu_time:
                break

        return model

    def one_step(self, model, transactions, level):
        raise NotImplementedError('Subclass responsibility')

    def custom_initial_solution(self, model, transactions):
        return model
