import numpy as np


def read_turbine_site_list(filename):
    with open(filename) as fid:
        lines = fid.readlines()
    turbine_model_name = lines[0].strip()
    data = np.array([l.replace(",", ".").split()[:4] for l in lines[1:] if l.strip() != ""])
    turbine_ids = data[:, 0]
    turbine_positions = data[:, 1:].astype(np.float)
    return turbine_model_name, turbine_ids, turbine_positions


def read_MR_turbine_site_list(filename):
    with open(filename) as fid:
        lines = fid.readlines()
    farm_name = lines[0].strip()
    nPairs = int(lines[1])

    nacelle_models = ([(lines[2 + 2 * i].strip(), list(map(float, lines[2 + 2 * i + 1].split()))) for i in range(nPairs)])
    turbine_positions = np.array([l.replace(",", ".").split()[:4] for l in lines[2 + 2 * nPairs:] if l.strip() != ""]).astype(np.float)
    return farm_name, nacelle_models, turbine_positions

