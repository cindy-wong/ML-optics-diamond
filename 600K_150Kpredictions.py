#%%
import pandas as pd
from mldtemp import *
import numpy as np
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib import cm
import pickle
import scipy.stats as sps
from dscribe.descriptors import MBTR
from ase.io.vasp import read_vasp
import os
import scipy.stats as sps
#%%
def mbtr(directory_path):

    poscar_dictionary = {}
    poscar_group = []
    mbtr_k2_grid_min = -0.1
    mbtr_k2_grid_max = (
        0.4  # 2.56 A interatomic distance ; ~0.4 is the inverse interatomic distance
    )

    mbtr_k2_grid_n = 50
    mbtr_k2_grid_sigma = 0.02
    mbtr_k2_weight_scale = 0.7
    mbtr_k2_weight_cutoff = 1e-3
    mbtr_materials = []
    name_log = []
    poscar_directory = os.listdir(directory_path)
    mbtr = MBTR(
        species=["C"],
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {
                "min": mbtr_k2_grid_min,
                "max": mbtr_k2_grid_max,
                "sigma": mbtr_k2_grid_sigma,
                "n": mbtr_k2_grid_n,
            },
            "weighting": {
                "function": "exp",
                "scale": mbtr_k2_weight_scale,
                "cutoff": mbtr_k2_weight_cutoff,
            },
        },
        periodic=True,
    )
    x_axis = mbtr.get_k2_axis()
    for subdirposcar, dirsposcar, filesposcar in sorted(os.walk(directory_path)):
        for poscar in filesposcar:
            if not poscar.startswith(".") and (poscar.startswith("POSCAR")):

                poscar_open = os.path.join(
                    subdirposcar, poscar
                )  # finds POSCAR from subdirectory
                # poscar_read=(ase.io.read(poscar_open, format="vasp"))
                poscar_read = read_vasp(
                    poscar_open
                )  # creates ASE structure with POSCAR file
                poscar_file = open(poscar_open, "r")  # opens POSCAR
                mbtr_element = pd.DataFrame(mbtr.create(poscar_read), columns=x_axis)
                name_log.append(poscar_open)
                mbtr_materials.append(mbtr_element)
    mbtr_df = pd.concat(mbtr_materials)
    return mbtr_df


#%%
mbtr_600K = mbtr("../snapshots_250_600K_50/")
mbtr_150K = mbtr("../snapshots_500_150K_50/")

mbtr_600K.to_csv("data/mbtr_600K.csv")
mbtr_150K.to_csv("data/mbtr_150K.csv")

#%%
mbtr_600K = pd.read_csv("data/mbtr_600K.csv", index_col=[0])
mbtr_150K = pd.read_csv("data/mbtr_150K.csv", index_col=[0])

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
SAVEFIGS = True

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

descriptor_data, eps_data = read_stats("./data")


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


#%% combine 150K and 600K into one dataframe
train_temps = temps
mbtr_150K.index = pd.MultiIndex.from_product([["150K"], list(range(len(mbtr_150K)))])
mbtr_600K.index = pd.MultiIndex.from_product([["600K"], list(range(len(mbtr_600K)))])
mbtr_150K.columns = descriptor.columns
mbtr_600K.columns = descriptor.columns
descriptor = pd.concat([descriptor, mbtr_150K, mbtr_600K])

#%% train model
d = 25
rs = 20
with open(f"pickles/bootstrap{d}_500.pkl", "rb") as f:
    results = pickle.load(f)
    scores = [(conf, np.mean(list(results[conf]))) for conf in results]
    scores = sorted(scores, key=lambda x: x[1])
    best_MSE = scores[0][1]
    best_conf = scores[0][0]
    samples = [int(int(s) * 250 / d) for s in best_conf[1:-1].split()]
    print(f"{d}: {samples}")
    best_conf = [int(i) for i in best_conf[1:-1].split()]
model = train_model(descriptor, eps, train_temps, best_conf, xgb_model, rs)

with open(f"pickles/bootstrap{d}_trained_model_rs{rs}.pkl", "wb") as f:
    pickle.dump(model, f)

#%%
with open(f"pickles/bootstrap{d}_trained_model_rs{rs}.pkl", "rb") as f:
    model = pickle.load(f)

#%% plot predictions with cis
all_temps = [f"{t}K" for t in sorted(list({int(i[0][:-1]) for i in descriptor.index}))][
    :-2
]
colors = cm.get_cmap("magma")(np.linspace(0, 1, len(all_temps) + 2))[1:-1]
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


def confidence_intervals(confidence, values, means):
    low_ci = []
    high_ci = []
    for i, c in enumerate(values.T):
        lci, hci = sps.t.interval(confidence, len(c), loc=means[i], scale=sps.sem(c))
        low_ci.append(lci)
        high_ci.append(hci)
    return low_ci, high_ci


plt.figure()
for t in all_temps:
    y_test_pred = model.predict(descriptor.loc[t])
    plt.plot(
        x_vals[start:end], np.mean(y_test_pred, axis=0), label=t, color=color_dict[t]
    )
    lci, hci = confidence_intervals(0.99, y_test_pred, np.mean(y_test_pred, axis=0))
    plt.fill_between(
        x=x_vals[start:end], y1=lci, y2=hci, alpha=0.5, color=color_dict[t]
    )

plt.xlim(2, 6)
plt.xlabel("Energy (eV)")
plt.ylabel(r"$\epsilon_2$", fontsize=24)
plt.legend()
if SAVEFIGS:
    plt.savefig("figures/600K_150K_predictions.png", bbox_inches="tight")


#%% plot differences
plt.figure(figsize=(9, 12))
for i, t in enumerate(temps):
    y_test_pred = np.mean(model.predict(descriptor.loc[t]), axis=0)
    y_train = np.mean(eps.loc[t], axis=0)
    plt.plot(
        x_vals[start:end], np.abs(y_test_pred - y_train) + i * 0.015, label=t, color=color_dict[t]
    )

plt.xlim(2, 6)
plt.xlabel("Energy (eV)")
plt.ylabel("Absolute Difference", fontsize=24)
plt.yticks(np.arange(18)*0.005, np.tile(np.arange(3)*0.005, 6))
plt.legend()
if SAVEFIGS:
    plt.savefig(f"figures/preds_absolute_diff_separated_rs{rs}.png", bbox_inches="tight")


#%% plot differences with moving average
window_width = 10
plt.figure(figsize=(9, 12))
for i, t in enumerate(temps):
    y_test_pred = np.mean(model.predict(descriptor.loc[t]), axis=0)
    y_train = np.mean(eps.loc[t], axis=0)
    moving_avg = np.abs(y_test_pred - y_train).rolling(window=window_width).mean()
    plt.plot(
        x_vals[start:end], moving_avg + i * 0.015, label=t, color=color_dict[t]
    )

plt.xlim(2, 6)
plt.xlabel("Energy (eV)")
plt.ylabel("Absolute Difference", fontsize=24)
plt.yticks(np.arange(18)*0.005, np.tile(np.arange(3)*0.005, 6))
plt.legend()
if SAVEFIGS:
    plt.savefig(f"figures/preds_absolute_diff_separated_rolling_avg_rs{rs}.png", bbox_inches="tight")
