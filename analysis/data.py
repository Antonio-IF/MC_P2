"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from functions import EDA, DQR, CSM 


raw_data = pd.read_csv('Data/train-2.csv', low_memory=False)

raw_EDA = EDA(raw_data)
raw_EDA.perform_EDA()

DQR = DQR(raw_data)
clean_data = DQR.perform_clean()

clean_EDA = EDA(clean_data)
clean_EDA.perform_EDA()

accuracy = CSM(clean_data)
accuracy.apply_scoring()