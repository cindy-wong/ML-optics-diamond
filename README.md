### Machine learning code for *Temperature-dependent dielectric function of diamond from first-principles and machine learning models* 

This repo contains the machine learning code and data for *Temperature-dependent dielectric function of diamond from first-principles and machine learning models*. 

The code consists of four files. As presented in the paper, they are used in the following order:
1. `xgb_bayessearch.py` trains an XGBoost model on the training data and uses Bayesian optimization for hyperparameter tuning. The best set of hyperparamters is left in the file as a comment.
2. `onetemppreds.py` trains the model on individual temperatures and tests on all temperatures, used to create Figure 3 in the paper.
3. `bootstrap.py` contains the bootstrapping procedure used to find the optimal temperature snapshot distributions, and also creates Figure 4 in the paper.
4. `600K_150Kpredictions.py` tests a model trained on a temperature snapshot distribution found from the bootstrapping procedure on 600K and 150K snapshots, used to create Figure 6 in the paper.

Note that all these four files are intended to run in Python Interactive mode in VSCode (similar to Jupyter Notebooks), not as scripts.

The functions in `./mldtemp/mlcomp.py` are all helper functions.

`./data` consists of all data `.csv` files used throughout the codebase, in particular the dielectric functions, the MBTRs, and the MSEs from one temperature trainings.

`./pickles` consists of pickles of various Python objects created from the bootstrapping procedure, specifically the searched temperature snapshot distributions and their MSEs, trained models, and the MSE distributions for the best temperature snapshot distributions.

