# Bootstrapping procedure

#%%
import pandas as pd
from mldtemp import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mldtemp import *
from matplotlib import cm
import pickle
from math import comb
from functools import reduce
from operator import iconcat

#%%
descriptor, eps = read_data("./data")
temps = [f"{t}K" for t in sorted(list({int(i[0][:-1]) for i in eps.index}))]
temps = temps[:-2]
starteV = 2
endeV = 6
x_vals = np.array([float(x) for x in eps.columns])
start = max((np.nonzero(np.array(x_vals) > starteV)[0][0]) - 1, 0)
end = min((np.nonzero(np.array(x_vals) < endeV)[0][-1]) + 2, len(x_vals))
eps = eps.iloc[:, start:end]
color_dict = {
    "700K": "#67001F",
    "600K": "#931D2C",
    "500K": "#C1373A",
    "400K": "#F09B7A",
    "300K": "#87BEDA",
    "200K": "#3079B6",
    "150K": "#1C538A",
    "100K": "#053061",
}

# xgboost model parameters found from bayesian search
xgb_model = MultiOutputRegressor(
    XGBRegressor(
        colsample_bytree=1.0,
        gamma=0.0,
        learning_rate=0.07317293050243988,
        max_depth=4,
        min_child_weight=4.790090733475141,
        n_estimators=314,
        subsample=0.43702421439040756,
    )
)

# train model on a specific fractional distribution of temperatures
def train_model(descriptor, eps, temps, snap_fracs, model, rs):
    assert len(temps) == len(snap_fracs)
    snap_fracs = np.array(snap_fracs) / sum(snap_fracs)
    X_train = shuffle(
        pd.concat(
            [
                descriptor.loc[t].sample(frac=f, random_state=rs)
                for t, f in zip(temps, snap_fracs)
            ]
        ),
        random_state=rs,
    )
    y_train = shuffle(
        pd.concat(
            [
                eps.loc[t].sample(frac=f, random_state=rs)
                for t, f in zip(temps, snap_fracs)
            ]
        ),
        random_state=rs,
    )
    model = model.fit(X_train, y_train)
    return model


# generates all lists of numbers of length l that sum to d
def gen_lists(d, l, vec, idx, sum):
    if sum == d:
        return [vec]
    if idx == l:
        return False

    out = []
    for i in range(d - sum + 1):
        vec[idx] += i
        if lst := gen_lists(d, l, vec.copy(), idx + 1, sum + i):
            out.append(lst)
        vec[idx] -= i
    return reduce(iconcat, out, [])


def generate_configs(d, N, l):
    """
    d: total sum of numbers
    N: number of vectors to generate
    l: length of vectors
    """
    out = np.array(gen_lists(d, l, [0] * l, 0, 0))
    total = comb(d + l - 1, l - 1)
    return out if total <= N else out[np.random.permutation(total)[:N]]


#%% 
# bootstrapping procedure

d = 15
N = 500
configs = generate_configs(d, N, 6)
best_MSE = np.inf
best_conf = None
rs = 0

results = {}
for c in configs:
    mses = np.zeros(len(temps))
    print(f"fitting {c} with rs={rs}...")
    m = train_model(descriptor, eps, temps, c, xgb_model, rs)
    print(f"predicting {c} with rs={rs}...")
    for i, t in enumerate(temps):
        X_test = descriptor.loc[t]
        y_test = eps.loc[t]
        y_test_pred = pd.DataFrame(
            index=y_test.index, columns=y_test.columns, data=m.predict(X_test)
        )
        mses[i] = mean_squared_error(y_test, y_test_pred)
    results[str(c)] = mses

#%% dump
with open("pickles/bootstrap10_500.pkl", "wb") as f:
    pickle.dump(results, f)

#%% load
with open(f"pickles/bootstrap{d}_1250_no700K.pkl", "rb") as f:
    results = pickle.load(f)

#%%
descriptor_data, eps_data = read_stats("./data")
ci_mses = {}
for t in temps:
    ci_mses[t] = mean_squared_error(
        eps_data.loc[("mean", t)][start:end], eps_data.loc[("lci99", t)][start:end]
    )

#%%
# sort temperature distributions by best ratio to confidence interval mses
scores = [
    (conf, np.prod([c / ci_mses[t] for t, c in zip(ci_mses, results[conf])]))
    for conf in results
]
scores = sorted(scores, key=lambda x: x[1])
best_MSE = scores[0][1]
best_conf = scores[0][0]

#%%
# sort temperature distributions by best average (used in paper)
scores = [(conf, np.mean([c for c in results[conf]])) for conf in results]
scores = sorted(scores, key=lambda x: x[1])
best_MSE = scores[0][1]
best_conf = scores[0][0]


#%%
# train model on specific number of samples per temperature
def train_model_num_samples(descriptor, eps, temps, samples, model, rs):
    X_train = shuffle(
        pd.concat(
            [
                descriptor.loc[t].sample(n=s, random_state=rs)
                for t, s in zip(temps, samples)
            ]
        ),
        random_state=rs,
    )
    y_train = shuffle(
        pd.concat(
            [eps.loc[t].sample(n=s, random_state=rs) for t, s in zip(temps, samples)]
        ),
        random_state=rs,
    )
    model = model.fit(X_train, y_train)
    return model

n_T = int(np.sqrt(250))
samples = [int(np.sqrt(int(s) / d * 250)) for s in best_conf[1:-1].split()] + [0]
distr = []

# train model on different random seeds to generate histogram of mses from temperature distribution
rs = range(np.prod([s for s in samples if s]))
for r in rs:
    print(f"fitting with rs={r}...")
    m = train_model_num_samples(descriptor, eps, temps, samples, xgb_model, r)
    mses = np.zeros(len(temps))
    print(f"predicting with rs={r}...")
    for i, t in enumerate(temps):
        X_test = descriptor.loc[t]
        y_test = eps.loc[t]
        y_test_pred = pd.DataFrame(
            index=y_test.index, columns=y_test.columns, data=m.predict(X_test)
        )
        mses[i] = mean_squared_error(y_test, y_test_pred)
    distr.append(np.mean(mses))

# dump
with open(f"pickles/d{d}_distr.pkl", "wb") as f:
    pickle.dump(distr, f)

#%% plot histograms
SAVEFIGS = True
for d in [15, 20, 25]:
    plt.figure()
    with open(f"pickles/d{d}_distr.pkl", "rb") as f:
        distr = pickle.load(f)
    with open(f"pickles/d{d}_distr_no700K.pkl", "rb") as f:
        no_700_distr = pickle.load(f)
    plt.axvline(x=np.mean(distr), color=cm.get_cmap("magma")(0.25),alpha=0.35)
    plt.axvline(x=np.mean(no_700_distr), color=cm.get_cmap("magma")(0.75),alpha=0.5)
    plt.hist(
        distr,
        bins=50,
        alpha=1,
        density=True,
        label="100K-500K and 700K",
        color="#8E65A9",
    )
    plt.hist(
        no_700_distr,
        bins=50,
        alpha=1,
        density=True,
        label="100K-500K",
        color=cm.get_cmap("magma")(0.8),
    )
    plt.hist(
        distr,
        bins=50,
        density=True,
        color="#8E65A9",
        fill=False,
        histtype='step',
    )   
    plt.xlabel("Average MSE")
    plt.ylabel("Density")
    plt.legend()
    if SAVEFIGS:
        plt.savefig(f"figures/d{d}_distr_no700K_hist.png")


#%%
def plot_cis(lci_df, hci_df):
    plt.figure()
    for t in temps:
        plt.fill_between(
            x=x_vals[start:end],
            y1=lci_df.loc[t][start:end],
            y2=hci_df.loc[t][start:end],
            alpha=0.5,
            color = color_dict[t],
            label=t
        )
    plt.legend()


# plots a single example of predictions based on a temperature distribution
def plot_predictions(c, descriptor, eps, model, train_model, rs=0):

    model = train_model(descriptor, eps, temps, c, model, rs)

    plot_cis(eps_data.loc["lci99"], eps_data.loc["hci99"])
    for t in temps:
        y_test_pred = model.predict(descriptor.loc[t])
        print(y_test_pred.shape)
        plt.plot(x_vals[start:end], np.mean(y_test_pred, axis=0))
        plt.xlim(2, 6)
        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$\epsilon_2$", fontsize=24)


for d in [20, 25]:
    with open(f"pickles/bootstrap{d}_500.pkl", "rb") as f:
        results = pickle.load(f)
    scores = [(conf, np.mean(list(results[conf]))) for conf in results]

    scores = sorted(scores, key=lambda x: x[1])
    best_MSE = scores[0][1]
    best_conf = scores[0][0]
    samples = [int(int(s) * 250 / d) for s in best_conf[1:-1].split()]
    print(f"{d}: {samples}")
    best_conf = [int(i) for i in best_conf[1:-1].split()]
    plot_predictions(best_conf, descriptor, eps, xgb_model, train_model, 0)

    if SAVEFIGS:
        plt.savefig(f"figures/d{d}predictions")
