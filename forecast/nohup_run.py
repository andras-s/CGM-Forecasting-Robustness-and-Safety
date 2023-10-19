import os
import sys
# os.system("nohup sh -c '" + sys.executable +
          # " main.py >results/nohup/LSTM_NLLPEG_16_05_extensive_logging.txt ' &"
          # " train_fes.py >results/nohup/FE_LSTM_0.txt ' &"
          # " train_ensembles.py >results/nohup/Ensembles_LSTM_FT_0.txt ' &"
          # " train_layers.py >results/nohup/Layers_LSTM_OLSvsL1_seed0.txt ' &"
          # " main.py >results/nohup/ConvT_NLLPEG0.015625_h1_baseline.txt ' &"
          # )

# os.system("nohup sh train_layers_lambdas_script.sh &")
os.system("nohup sh train_layers_loop.sh &")
