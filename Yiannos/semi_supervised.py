import pickle
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

df = pd.read_csv('test_set_220727.csv')
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df = df.loc[df["animal"] != 'CW_GP_190826']  # remove data from animal CW_GP_190826 because of weird bf
df_new = df
df_new = df_new.drop(columns=['animal', 't', 'ch', 'isi_cv',  'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r', 'bf','sync_n', 'bf_deviation', 'sfr', 'r_max'])  # dropped  # dropped

#   'd', 'd_max',  not dropped

df_unlabeled = df_new.dropna()
df_unlabeled.loc[:, 'group'] = -1
y_unlabeled = df_unlabeled.loc[:, "group"]
df_unlabeled = df_unlabeled.drop(columns=['group'])
X_unlabeled = StandardScaler().fit_transform(df_unlabeled)

df1 = pd.read_csv('train_set.csv')
df1 = df1.loc[df1["group"] != "ET_T"]  # remove ET_T data-points
df1 = df1.loc[df1["group"] != "ET_A"]  # remove ET_T data-points
df1 = df1.loc[df1["group"] != "NE"]  # remove NE data-points
df1 = df1.loc[df1["ch"] <= 31]  # remove data from channels past 31
df1["group"].replace({"ET_E": 'ET'}, inplace=True)  # mix data-points with ET
df1["group"].replace({"ET": 2}, inplace=True)
df1["group"].replace({"ENT": 1}, inplace=True)
#df1["group"].replace({"NE": 0}, inplace=True)

df1_new = df1
df1_new = df1_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r', 'bf', 'sync_n', 'bf_deviation', 'r_max', 'sfr'])  # dropped

#   'd_max', 'd', group  not dropped

df1_new = df1_new.dropna()
y_labeled = df1_new.loc[:, "group"]
df1_new = df1_new.drop(columns=['group'])
X_labeled = StandardScaler().fit_transform(df1_new)
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2)

model = MLPClassifier(solver='lbfgs', max_iter=300, alpha=0.001)
X = np.concatenate((np.array(X_train), np.array(X_unlabeled)))
y = np.concatenate((np.array(y_train), np.array(y_unlabeled)))

self_training_model = SelfTrainingClassifier(model)
self_training_model.fit(X, y)
print(self_training_model.score(X_test, y_test))

pickle.dump(self_training_model, open('models/mlp_self_training_binary_d_dmax', 'wb'))


