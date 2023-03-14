import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor


def RandomCatBoostClassifier(seed=1, custom_params = None):
    """
    Return a random CatBoost classifier
    """
    param_grid = {'depth': np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                  'bagging_temperature': np.random.choice([0.1, 0.2, 0.3,
                                                           0.4, 0.5, 0.6,
                                                           0.7, 0.8, 0.9,
                                                           1.0, 2, 3,
                                                           5, 10]),
                  'learning_rate': np.random.choice([0.01, 0.01, 0.01,
                                                     0.02, 0.03, 0.04,
                                                     0.05, 0.06, 0.07, 
                                                     0.08, 0.09, 0.1,
                                                     0.1, 0.1, 0.15,
                                                     0.2]),
                  'colsample_bylevel': np.random.choice([0.3, 0.4, 0.5,
                                                         0.5, 0.6, 0.6,
                                                         0.7, 0.7, 0.8,
                                                         0.8, 0.9, 1.0]),
                  'l2_leaf_reg': np.random.choice([0.01, 0.5, 1,
                                                   2, 3, 4,
                                                   5, 6, 7,
                                                   8, 9, 10,
                                                   30, 100]),
                  'iterations': np.random.choice([10, 20, 30,
                                                  40, 50, 100,
                                                  250, 500, 1000]),
                  'random_strength': np.random.uniform(0, 100),
                  'verbose': False,
                  'bootstrap_type': np.random.choice(['Bayesian'])
                 }
    
    if custom_params:
        for k in custom_params.keys():
            try:
                param_grid[k] = np.random.choice(custom_params[k])
            except:
                param_grid[k] = custom_params[k][0]
    
    clf = CatBoostClassifier(**param_grid)
    return clf

class RandomSearchCatBoost:
    """
    Performs a random search to find the best hyperparameters for CatBoost Models

    Attributes:
    num_iterations (int): integer with the number of iterations to run of the algorithm.
    tipo (string): classifier or regressor to choose the right model to test.
    results (dict): Dictionary with the parameters used and score
    best_score (float): Score from the best model.
    best_params (dict): Parameters of best model.

    """
    def __init__(self,
                 num_iterations=200,
                 tipo='classifier',
                 seed = 1,
                 custom_params=None,
                 verbose: bool=True):
        self.num_iterations = num_iterations
        self.tipo = tipo
        self.results = {}
        self.best_score = 0
        self.best_params = {}
        self.model = object
        self.seed = seed
        self.custom_params=custom_params
        self.verbose = verbose

    def fit(self,
            X_train, y_train,
            X_val, y_val):
        seed = self.seed
        for i in range(self.num_iterations):
            if self.tipo == 'classifier':
                model_inst = RandomCatBoostClassifier(seed, custom_params=self.custom_params)
            elif self.tipo == 'regressor':
                model_inst = RandomCatBoostRegressor(seed, custom_params=self.custom_params)
            
            auc_cv = cross_val_score(model_inst,
                            X_train,
                            y_train,
                            cv=5,
                            scoring='roc_auc').mean()
            model_inst.fit(X_train, y_train)
            p = model_inst.predict_proba(X_val)[:, 1]
            auc_test = metrics.roc_auc_score(y_val, p)
            
            if self.verbose:
                print("Iteration:", i)
                print("AUC CV:", auc_cv)
                print("AUC Test:", auc_test)
                print("Params:")
                print(str(model_inst.get_params()))
                print()
                
            self.results[i] = {"model_params": model_inst.get_params(),
                              "auc_cv": auc_cv,
                              "auc_test": auc_test,
                             }

            if auc_cv > self.best_score:
                self.best_score = auc_cv
                self.best_params = model_inst.get_params()
                self.best_iteration = i

            seed += 1
        if self.tipo == 'classifier':
            self.model = CatBoostClassifier(**self.best_params)
        elif self.tipo == 'regressor':
            self.model = CatBoostRegressor(**self.best_params)

        self.model.fit(X_train, y_train)
    """
    Fits a random search to find the best hyperparameters for Catboost Models

    Args:
    X_train (pd.DataFrame or numpy.array): pd.DataFrame or numpy.array with all features for the train dataset.
    y_train (pd.Series or numpy.array): pd.Series or numpy.array with the target for train dataset.
    X_val (pd.DataFrame or numpy.array): pd.DataFrame or numpy.array with all features for the validation dataset.
    y_val (pd.Series or numpy.array): pd.Series or numpy.array with the target for validation dataset.

    Returns:
    Fitted model. Parameters can will be saved in class atributes
    """

    def predict_proba(self, X_val):
        p = self.model.predict_proba(X_val)
        return p

    def predict(self, X_val):
        p = self.model.predict(X_val)
        return p

# Modo de uso, dado X_train, X_test já separados e processados testar neles
    
# rscb = RandomSearchCatBoost(num_iterations=30,
#                             verbose=True)

# rscb.fit(X_train=X_train,
#          y_train=y_train,
#          X_val=X_test,
#          y_val=y_test)

# rscb.results.keys()