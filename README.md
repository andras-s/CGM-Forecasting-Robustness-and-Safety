# CGM-Forecasting-Robustness-and-Safety
This repo is the result of my thesis "Enhancements in CGM Forecasting: Robustness Against Domain Shifts and Safer Predictions". It analyzes and improves the robustness against domain shifts and the clinical safety of predictions made by state-of-the-art deep learning architectures in CGM forecasting.

Before running this code, please make sure the data paths are adjusted appropriately. Most importantly, the constant DATA_DIR from utils/setup.py should be changed.

All python scripts should be run from the root directory of the repository, and can be run either from the terminal, or from ipython/jupyter notebook (adjust setup in forecast/config.py and impute/config.py to run the scripts from a bash script).
