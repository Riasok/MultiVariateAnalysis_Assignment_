import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
from sklearn.metrics import confusion_matrix, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS
import time


#Read dataset
df_i = pd.read_csv("C:/Users/kosai/Desktop/다변량2/high_diamond_ranked_10min.csvhigh_diamond_ranked_10min.csv")
print(df_i.head())

