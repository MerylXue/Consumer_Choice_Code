# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.




class MarketExplorer(object):
    def explore_for(self, estimator, model, transactions):
        raise NotImplementedError('Subclass responsibility')
