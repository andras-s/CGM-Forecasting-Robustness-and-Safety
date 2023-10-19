import os
import json

import pandas as pd
import numpy as np

from itertools import product


peg = False      # for GDU results set this to false todo: adapt the gdu file structure in FileZilla (Copy in unified folders, which are currently empty)

if peg:
    models = ["ConvT", "LSTM"]
    losses = ["NLL", "NLLPEGSurface"]
    run_types = [f"{model}_{loss}" for model, loss in product(models, losses)]
    results_path = "/local/home/ansass/Thesis/icarus/analysis/Discussion/runtimes/"
else:
    run_types = ["ERM_single", "ERM_ensemble", "FT_CS", "FT_MMD", "E2E_CS", "E2E_MMD"]
    results_path = "/local/home/ansass/Thesis/icarus/analysis/Discussion/runtimes/GDU/"

results = {"run_type": [],
           "avg runtime per epoch": [],
           "std runtime per epoch": [],
           "avg num epochs": [],
           "std num epochs": [],
           "avg total runtime": [],
           "std total runtime": [],
           }
for run_type in run_types:
    run_path = os.path.join(results_path, f"{run_type}")
    runtimes = {}
    for fold in [0, 1, 2, 3, 4]:
        with open(os.path.join(run_path, f"runtimes_fold_{fold}.txt")) as file:
            runtimes[fold] = json.load(file)
    all_runtimes = []
    [all_runtimes.extend(runtimes[k]) for k in runtimes.keys()]
    nums_epochs = [len(runtimes[k]) for k in runtimes.keys()]
    total_runtimes = [sum(runtimes[k]) for k in runtimes.keys()]
    results["run_type"].append(run_type)
    results["avg runtime per epoch"].append(np.mean(all_runtimes))
    results["std runtime per epoch"].append(np.std(all_runtimes))
    results["avg num epochs"].append(np.mean(nums_epochs))
    results["std num epochs"].append(np.std(nums_epochs))
    results["avg total runtime"].append(np.mean(total_runtimes))
    results["std total runtime"].append(np.std(total_runtimes))

results_df = pd.DataFrame.from_dict(results)
print(results_df)

print(
    results_df.to_latex(index=False, formatters={"name": str.upper}, float_format="{:.2f}".format)
)
