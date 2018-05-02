from sklearn.base import TransformerMixin, BaseEstimator


class FactorExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,factor):
        if factor is not None:
            self.factor=factor
        else:
            self.factor=None
    
    def transform(self, data):
        if self.factor is None:
            return data
        else:
            return data[self.factor]

    def fit(self,*_):
        return self
        



