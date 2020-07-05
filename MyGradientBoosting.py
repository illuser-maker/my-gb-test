import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


class MyMSELoss:
    '''Loss object for MyGradientBoostingRegressor.'''
    def __init__(self):
        pass
    
    def derivative(self, approxes, targets):
        '''Must return list of negative first derivatives. Params:
            approxes : numpy array, our predictions;
            targets : numpy array, true values.'''
        assert len(approxes) == len(targets)
        der1 = targets - approxes
        return der1    
        

class MyGradientBoostingRegressor:
    '''Class of custom Gradient Boosting Regressor. The realization is very simple. 
    Base algorithm: Decission Tree Regressor. Init params:
        metric : function, is used as metric(y_true, y_pred).
        loss : loss object, must support method derivative, returning minus first derivative.'''
    def __init__(self, metric=mean_squared_error, loss=MyMSELoss()):
        self.base_algo = DecisionTreeRegressor
        if not callable(metric):
            raise Exception("Metric must be func type")
        self.metric = metric
        self.loss = loss
        self.models = None

    def fit(self, X, y, iterations=10, learning_rate=0.1, max_depth=None, max_leaves=None, min_data_in_leaf=1, 
            verbose=None, eval_set=None):
        '''Fit boosting. Parameters:
            X : numpy array, dataset for fit;
            y : numpy vector, target variable;
            iterations : integer, number of trees/iterations;
            learning_rate : float,  learning rate coefficient;
            max_depth : integer or None, maximum depth of all trees;
            max_leaves : integer or None, maximum number of leaves in trees
            min_data_in_leaf : integer, minimum number of data rows in one leaf node of tree;
            verbose : boolean or integer, verbose output, integer means iteration for output;
            eval_set : list of format [X, y], used for output metric value in verbose mode.'''
        self.models = []
        self.target = y
        self.learning_rate = learning_rate
        cur_pred = 0
        eval_pred = 0
        eval_string = ""
        for i in range(iterations):
            self.models.append(self.base_algo(max_depth=max_depth, min_samples_leaf=min_data_in_leaf, 
                                              max_leaf_nodes=max_leaves))
            self.models[-1].fit(X, self.target)
            cur_pred += self.learning_rate * self.models[-1].predict(X)
            if eval_set:
                eval_pred += self.learning_rate * self.models[-1].predict(eval_set[0])
            self.target = self.loss.derivative(cur_pred, y)
            if verbose:
                if isinstance(verbose, bool) or (isinstance(verbose, int) and ((i + 1) % verbose == 0)):
                    if eval_set:
                        eval_string = f"\tEval metric:{self.metric(eval_set[1], eval_pred):>12.4f}"
                    print(f"Iteration: {i+1}\tTrain metric: {self.metric(y, cur_pred):>12.4f}" + eval_string)
        print(f"Final Metric: {self.metric(y, cur_pred)}")
        return self
    
    
    def predict(self, X, y=None):
        '''Predict target. Needs X - the same shape as in fitting.'''
        if self.models is None or self.models == []:
            raise Exception("Model is unfit")
        pred = 0
        for model in self.models:
            pred += self.learning_rate * model.predict(X)
        return pred
        
    def fit_predict(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.predict(X)
