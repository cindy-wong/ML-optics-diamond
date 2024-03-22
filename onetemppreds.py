# Training on one temperature and predicting on another

#%%
import pandas as pd
from mldtemp import *
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mldtemp import *

#%%
temps = [f"{str(s)}K" for s in [100, 200, 300, 400, 500, 700, 1000, 2000]]
descriptor, eps = read_data("./data")
x_vals = np.array([float(x) for x in eps.columns])

# dielectric between 2 and 6
starteV = 2
endeV = 6
start = max((np.nonzero(np.array(x_vals) > starteV)[0][0]) - 1, 0)
end = min((np.nonzero(np.array(x_vals) < endeV)[0][-1]) + 2, len(x_vals))
eps = eps.iloc[:, start:end]

SAVEFIGS = True
#%%
# hyperparameters from bayesian search
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

# trains model on a specific temperature and returns prediction on all temperatures
def test_temp(descriptor, eps, temps, train_temp, model, num_snapshots=None):
    if num_snapshots and num_snapshots != 250:
        X_train = descriptor.loc[train_temp].sample(num_snapshots)
    else:
        X_train = descriptor.loc[train_temp]
    y_train = eps.loc[train_temp].loc[X_train.index]
    model = model.fit(X_train, y_train)
    X_test = descriptor.loc[temps]
    y_test = eps.loc[temps]
    y_test_pred = pd.DataFrame(
        index=y_test.index, columns=y_test.columns, data=model.predict(X_test)
    )
    return y_test, y_test_pred

for n in [50, 100, 150, 200, 250]:
    mat = np.zeros((len(temps), len(temps)))
    for i in range(len(temps)):
        print(f"fitting {temps[i]}...")
        y_test, y_test_pred = test_temp(descriptor, eps, temps, temps[i], xgb_model, n)
        for j in range(len(temps)):
            mat[i, j] = mean_squared_error(
                y_test.loc[temps[j]][start:end], y_test_pred.loc[temps[j]][start:end]
            )
    df = pd.DataFrame(
        mat,
        index=pd.MultiIndex.from_product([["training temp"], temps]),
        columns=pd.MultiIndex.from_product([["testing temp"], temps]),
    )
df.to_csv(f"data/{n}onetemppreds.csv")


# %% one temp predictions error matrix in one plot
fig, axs = plt.subplots(3, 1, figsize=(9, 18))

vmin = 100
vmax = -1
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.minor.visible"] = "False"
plt.rcParams["ytick.minor.visible"] = "False"
for i, n in enumerate([50, 150, 250]):
    df = -np.log10(
        pd.read_csv(f"data/{n}onetemppreds.csv", index_col=[0, 1], header=[0, 1])
    )
    vmin = min(np.min(df.to_numpy()), vmin)
    vmax = max(np.max(df.to_numpy()), vmax)


for i, n in enumerate([50, 150, 250]):
    ax = axs[i]
    plt.sca(ax)
    df = pd.read_csv(f"data/{n}onetemppreds.csv", index_col=[0, 1], header=[0, 1])
    sns.heatmap(
        -np.log10(df),
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "-log(MSE)"},
        cmap="magma_r",
    )

    df_arr = df.to_numpy()
    dmax = np.max(df_arr)
    dmin = np.min(df_arr)
    temps = df.index.get_level_values(1)
    print(f"{n}: max: {dmax} at train {temps[np.where(df_arr==dmax)[0]]} test {temps[np.where(df_arr==dmax)[1]]} \t min: {dmin} at train {temps[np.where(df_arr==dmin)[0]]} test {temps[np.where(df_arr==dmin)[1]]}")

    plt.xlabel("Testing temperature")
    plt.ylabel("Training temperature")
    ax.set_xticklabels([idx[1] for idx in df.index])
    ax.set_yticklabels([idx[1] for idx in df.index])
    ax.tick_params(which="both", left=False, bottom=False)

    ax.set_title(f"({chr(ord('a') + i)})", ha='left', x=-0.1, y=1.05)
fig.tight_layout()
if SAVEFIGS:
    plt.savefig(
        f"figures/MSEheatmapneglog.png",
        bbox_inches="tight",
    )