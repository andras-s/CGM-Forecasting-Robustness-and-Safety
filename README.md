# CGM-Forecasting-Robustness-and-Safety
This repo is the result of my thesis "Enhancements in CGM Forecasting: Robustness Against Domain Shifts and Safer Predictions". It analyzes and improves the robustness against domain shifts and the clinical safety of predictions made by state-of-the-art deep learning architectures in CGM forecasting.

Before running this code, please make sure the data paths are adjusted appropriately. Most importantly, the constant DATA_DIR from utils/setup.py should be changed.


The investigation of the PEG loss’ and the GDU models’ influence on CGM forecasting resulted in a comprehensive codebase. It was developed to preprocess raw CGM data, train
and evaluate the models described, as well as analyse the results. Our implementation was developed in Python 3.9 and is modularized into three key components.

### Preprocessing
  This module loads the raw CGM dataset, executing the artifact handling methodologies elucidated in Section 2.1.2. Post-cleaning, the data is split—either into 5 folds at random or based on the intrinsic treatment classifications of the individuals. This processed data is subsequently saved, ready for the forecasting module’s consumption.
  
### Forecasting
  Centralized within this module are the model architectures, metric definitions, loss function formulations, and training protocols. Given the appropriate model architecture, hyperparameters, and data partitioning scheme, the system trains the models, subsequently computing the evaluation measures. Both the refined models and their evaluative performance metrics are stored for subsequent analysis.

### Analysis
  This module undertakes the systematic aggregation and comparison of performance metrics across diverse model trainings. It’s equipped to compute and represent distances
amid feature vectors and bases of GDU models. Moreover, it facilitates the generation of visual interpretations, showcasing exemplary forecasting outcomes, performance differences, and t-distributed stochastic neighbor embedding (t-SNE) embeddings.

Several libraries played crucial roles in the implementation of our methodologies. Pandas was essential for CGM data preprocessing, while NumPy handled array computations and metric calculations. Gaussian interpolation was handled by GPytorch while scikit-learn contributed to data preprocessing and t-SNE-based visualization of the GDU feature
vectors and bases. Deep learning architectures. metrics, and loss functions were realized with PyTorch, complemented by CUDA’s GPU acceleration on a NVIDIA’s RTX 3090. Finally, visual findings were illustrated using Matplotlib and Seaborn.
