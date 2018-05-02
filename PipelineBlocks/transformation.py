from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class FactorExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,factor):
   
        self.factor=factor
        
    
    def transform(self, data):

        return data[self.factor]

    def fit(self,*_):
        return self
        



class ModelTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name,model,eval_metric=None):
        self.name=name
        self.model=model
        self.eval_metric=eval_metric
    
    def fit(self,*args,**kwargs):
        self.model.fit(eval_metric=self.eval_metric,*args,**kwargs)
        return self

    def transform(self, X,**transform_params):
        return pd.DataFrame(self.model.predict(X),columns=[self.name+'_prediction'])
        