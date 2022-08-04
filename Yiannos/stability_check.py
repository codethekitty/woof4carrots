import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

df = pd.read_csv('test_set_220727.csv')
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df = df.loc[df["animal"] != 'CW_GP_190826']  # remove data from animal CW_GP_190826 because of weird bf
df_new = df
df_new = df_new.drop(columns=['ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n'])  # dropped

# , 'bf_deviation', 'bf', 'd_max', 'd'  not dropped

df_new = df_new.dropna()

tree = pickle.load(open('models/tree50.sav', 'rb'))
knn = pickle.load(open('models/knn50.sav', 'rb'))
svm = pickle.load(open('models/svm50.sav', 'rb'))

animals = df_new['animal'].unique()

res = []
xs = []
for animal in animals:
    res_t_tree = []
    res_t_knn = []
    res_t_svm = []
    df_a = df_new.loc[df_new["animal"] == animal]
    times = df_a['t'].unique()
    for time in times:
        df_t = df_a.loc[df_a["t"] == time]
        df_f = df_t.drop(columns=['animal', 't'])  # dropped
        X = StandardScaler().fit_transform(df_f)
        pred_tree = tree.predict(X)
        percent_t_tree = np.count_nonzero(np.array(pred_tree)) / len(pred_tree)
        res_t_tree.append(percent_t_tree)
    res.append(res_t_tree)
    xs.append(times)

plt.xlabel('Time', fontsize=10)
plt.ylabel('Percent Neurons Classified As Tinnitus', fontsize=10)
plt.title('Stability of Classification with Tree', fontsize=15)
for i in range(20):
    plt.plot(xs[i], res[i], label="{}".format(animals[i]))
plt.legend(loc='best')
plt.show()
