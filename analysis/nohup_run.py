import os
import sys
os.system("nohup sh -c '" +
          # sys.executable + " k_means_analysis.py >results/nohup/k_means_horizon_2_seed_4.txt ' &")
          sys.executable + " 13_tSNE.py' &")

# os.system("nohup sh 14_measure_runtimes_loop.sh &")
