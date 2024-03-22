# Trains an XGBoost model predicting dielectric function from MBTR snapshots
# with hyperparameter tuning done with Bayesian optimization

#%%
import skopt
import numpy as np
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from time import time
from skopt.space import Real, Categorical, Integer
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import pickle
from xgboost import XGBRegressor

#%%
descriptor = pd.read_csv("data/mbtr.csv", index_col=[0, 1])
eps = pd.read_csv("data/eps.csv", index_col=[0, 1])
engs = pd.read_csv("data/energies.csv", index_col=[0, 1])
x_vals = np.array([float(x) for x in eps.columns])
starteV = 2
endeV = 6
start = max((np.nonzero(np.array(x_vals) > starteV)[0][0]) - 1, 0)
end = min((np.nonzero(np.array(x_vals) < endeV)[0][-1]) + 2, len(x_vals))
eps = eps.iloc[:, start:end]
X_train, X_test, y_train, y_test = train_test_split(descriptor, eps, train_size=0.8)

#%%
grid = {
        'estimator__min_child_weight': Real(0, 30),
        'estimator__gamma': Real(0, 1000),
        'estimator__subsample': Real(0.4,1),
        'estimator__colsample_bytree': Real(0.4,1),
        'estimator__max_depth': Integer(3,8),
        'estimator__learning_rate': Real(1e-2,0.5,prior="log-uniform"),
        'estimator__n_estimators': Integer(50,500),
        }

xgb = MultiOutputRegressor(XGBRegressor())
opt = BayesSearchCV(
    xgb,
    grid,
    scoring='neg_mean_absolute_error',
    n_iter=120,
    cv=5,
    verbose=15,
    n_jobs=1
)

start = time()
# callback handler
checkpoint_callback = skopt.callbacks.CheckpointSaver("./xgb_2-6.pkl")

opt.fit(X_train, y_train, callback=[checkpoint_callback])

with open("xgb_2-6.pkl", "wb") as handle:
    pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"best score: {opt.best_score_}")
print(f"best params: {opt.best_params_}")
print(f"test score: {opt.score(X_test, y_test)}")
print(f"total time: {time()-start}")
