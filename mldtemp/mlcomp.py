from dscribe.descriptors import MBTR
from ase.io.vasp import read_vasp
import os
import pandas as pd
import numpy as np
from lxml import etree as ET
import scipy.stats as sps
from monty.io import zopen
from scipy.interpolate import interp1d


def mbtr(entries, api_client):
    mbtr_k2_grid_min = -0.1
    mbtr_k2_grid_max = 0.4
    mbtr_k2_grid_n = 50
    mbtr_k2_grid_sigma = 0.02
    mbtr_k2_weight_scale = 0.7
    mbtr_k2_weight_cutoff = 1e-3
    mbtr_materials = []
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
    with open("POSCAR", "w") as f1:
        with open("POTCAR", "w") as f2:
            for entry in entries:
                try:
                    raw = api_client.get_raw(entry, "POSCAR")
                    f1.write(raw)
                    f1.flush()  # for some reason this is necessary for a file without an extension
                    raw = api_client.get_raw(entry, "POTCAR")
                    f2.write(raw)
                    f2.flush()
                    poscar_open = os.path.join(os.getcwd(), "POSCAR")
                    poscar_read = read_vasp(poscar_open)
                    mbtr_element = pd.DataFrame(
                        mbtr.create(poscar_read), columns=x_axis
                    )
                    mbtr_materials.append(mbtr_element)
                    f1.truncate(0)  # clear contents of file
                    f1.seek(0)
                    f2.truncate(0)
                    f2.seek(0)
                except Exception as exc:
                    print(f"{entry['mainfile']} failed. error: {exc}")
                    mbtr_materials.append(
                        pd.DataFrame(np.full((1, len(x_axis)), np.nan), columns=x_axis)
                    )
    os.remove(poscar_open)
    os.remove(os.path.join(os.getcwd(), "POTCAR"))
    return pd.concat(mbtr_materials)


# dielectric parsing function from pymatgen
def _parse(file):
    with zopen(file, "rb") as f:
        for _, elem in ET.iterparse(f):
            if elem.tag == "energy":
                for e in elem.findall("i"):
                    if e.attrib["name"] == "e_0_energy":
                        eng = float(elem.find("i").text)
            if elem.tag == "dielectricfunction":
                imag = [
                    [float(l) for l in r.text.split()]
                    for r in elem.find("imag").find("array").find("set").findall("r")
                ]
                elem.clear()
                return [e[1:] for e in imag], [e[0] for e in imag], eng


def dielectric_function(entries, api_client, engs=None):
    eps_imag_all = []
    energies = np.zeros(len(entries))
    with open("vasprun_temp.xml", "w") as f:
        for i, entry in enumerate(entries):
            try:
                f.write(api_client.get_raw(entry))
                e2, engs_all, f_eng = _parse("vasprun_temp.xml")
                e2 = np.array(e2)
                energies[i] = f_eng
                e2x = e2[:, 0]
                e2y = e2[:, 1]
                e2z = e2[:, 2]
                etoti = (e2x + e2y + e2z) / 3
                if engs is None:
                    # full dielectric
                    eps_imag = pd.DataFrame(etoti, index=engs_all).T
                elif len(engs) == 2:
                    # dielectric in between engs[0] and engs[1]
                    start = max((np.nonzero(np.array(engs_all) > engs[0])[0][0]) - 1, 0)
                    end = min(
                        (np.nonzero(np.array(engs_all) < engs[1])[0][-1]) + 2,
                        len(engs_all),
                    )
                    eps_imag = pd.DataFrame(
                        etoti[start:end], index=engs_all[start:end]
                    ).T
                else:
                    # dielectric at energies specified by engs
                    interp = interp1d(engs_all, etoti)
                    eps_imag = pd.DataFrame(interp(engs), index=engs).T
                eps_imag_all.append(eps_imag)
                print(f"{entry['mainfile']} success. ")
            except Exception as exc:
                print(f"{entry['mainfile']} failed. error: {exc}")
                cols = eps_imag_all[0].columns
                eps_imag_all.append(
                    pd.DataFrame(np.full((1, len(cols)), np.nan), columns=cols)
                )
            f.truncate(0)  # clear contents of file
            f.seek(0)
    os.remove(os.path.join(os.getcwd(), "vasprun_temp.xml"))
    return pd.concat(eps_imag_all), energies


def _confidence_intervals(confidence, values, means):
    low_ci = []
    high_ci = []
    for i, c in enumerate(values.T):
        lci, hci = sps.t.interval(confidence, len(c), loc=means[i], scale=sps.sem(c))
        low_ci.append(lci)
        high_ci.append(hci)
    return low_ci, high_ci


def stats(descriptor, eps):
    idx = pd.MultiIndex.from_product(
        [
            [
                "mean",
                "std",
                "lci95",
                "hci95",
                "ci95_width",
                "lci99",
                "hci99",
                "ci99_width",
                "CV",
            ],
            ["all"] + sorted(list({i[0] for i in eps.index})),
        ]
    )
    descriptor_stats = pd.DataFrame(index=idx, columns=descriptor.columns)
    eps_stats = pd.DataFrame(index=idx, columns=eps.columns)
    for s, d in [(eps_stats, eps), (descriptor_stats, descriptor)]:
        for c in sorted({i[0] for i in eps.index} | {"all"}):
            data = d.to_numpy() if c == "all" else d.loc[c].to_numpy()
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            s.loc[("mean", c), :] = mean
            s.loc[("std", c), :] = std
            s.loc[("CV", c), :] = std / mean
            for conf in ["95", "99"]:
                (
                    s.loc[("lci" + conf, c), :],
                    s.loc[("hci" + conf, c), :],
                ) = _confidence_intervals(int(conf) / 100, data, s.loc[("mean", c), :])
                s.loc[("ci" + conf + "_width", c), :] = (
                    s.loc[("hci" + conf, c), :] - s.loc[("lci" + conf, c), :]
                )
        s.fillna(0, inplace=True)
    return descriptor_stats, eps_stats


def read_data(folder):
    return (
        pd.read_csv(f"{folder}/mbtr.csv", index_col=[0, 1]),
        pd.read_csv(f"{folder}/eps.csv", index_col=[0, 1]),
    )


def read_stats(folder):
    return (
        pd.read_csv(f"{folder}/mbtr_stats.csv", index_col=[0, 1]),
        pd.read_csv(f"{folder}/eps_stats.csv", index_col=[0, 1]),
    )


def write_stats(descriptor, eps, folder):
    descriptor_stats, eps_stats = stats(descriptor, eps)
    eps_stats.to_csv(f"{folder}/eps_stats.csv")
    descriptor_stats.to_csv(f"{folder}/mbtr_stats.csv")