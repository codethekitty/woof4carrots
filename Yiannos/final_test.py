import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics

df = pd.read_csv('mike_data.csv')
df["group"].replace({"ET": 2}, inplace=True)
df["group"].replace({"ENT": 1}, inplace=True)
df["group"].replace({"NE": 0}, inplace=True)

df_new = df
df_new = df_new.drop(columns=['animal', 'ch', 'ind', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'isi_cv', 'br', 'r', 'd_max', 'd'])  # dropped

#  'sfr', 'r_max','bf', 'bf_deviation', 'sync_n', 'group' not dropped

df_new = df_new.dropna()
y = df_new.loc[:, "group"]
df_new = df_new.drop(columns=['group'])
X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y)

model = pickle.load(open('models/tree_ovo', 'rb'))

print(model.score(X, y))
