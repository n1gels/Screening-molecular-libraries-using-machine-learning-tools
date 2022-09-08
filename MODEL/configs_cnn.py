# Analyse descriptor structure
import numpy as np
desc = np.genfromtxt(r'DATA/desc.csv', dtype=str)
c_d = len(desc[0]) - 1  # 3
site = np.genfromtxt('DATA/site_desc.csv', dtype=str)
c_s = len(site[0]) - 2  # 3
c_m = 2 * c_d + c_s  # 9
c_i = 2 * c_m  # 18
n_outcomes = 1
n_features = c_i  # 18
n_units = 40
dropout = 0.1
