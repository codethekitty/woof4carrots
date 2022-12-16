import pickle
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics

df = pd.read_csv('mike_data.csv')
df["group"].replace({"ET": 2}, inplace=True)
df["group"].replace({"ENT": 1}, inplace=True)
df["group"].replace({"NE": 0}, inplace=True)

df_new = df
df_new = df_new.drop(columns=['ch', 'ind', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'isi_cv', 'br', 'sfr', 'r', 'r_max', 'sync_n', 'bf_deviation', 'bf'])  # dropped

#  'd_max', 'd'  'group', animal not dropped

df_new = df_new.dropna()
y = df_new.loc[:, "group"]
df_new = df_new.drop(columns=['group'])
#X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y)

model = pickle.load(open('models/knn_binary_d_dmax_uniform', 'rb'))

animals = df_new['animal'].unique()
res = []

for animal in animals:
    df_a = df_new.loc[df_new["animal"] == animal]
    df_f = df_a.drop(columns=['animal'])  # dropped
    X = StandardScaler().fit_transform(df_f)
    pred = model.predict(X)

    prob_t = np.count_nonzero(np.array(pred) == 2) / len(pred)
    prob_ne = (len(np.array(pred)) - np.count_nonzero(np.array(pred))) / len(pred)
    prob_ent = np.count_nonzero(np.array(pred) == 1) / len(pred)

    print(animal)
    print(prob_t)

    if prob_t >= 0.5:
        res.append(2)
    else:
        res.append(1)

print(res)
print([2, 1, 2, 2, 2, 2, 1, 1, 1])

