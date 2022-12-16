import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

df = pd.read_csv('test_set_220727.csv')
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df = df.loc[df["animal"] != 'CW_GP_190826']  # remove data from animal CW_GP_190826 because of weird bf
df_new = df
df_new = df_new.drop(columns=['ch', 'isi_cv', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r', 'sfr', 'br', 'bf_deviation', 'bf', 'sync_n', 'r_max'])  # dropped

#  'd_max', 'd'   not dropped

df_new = df_new.dropna()

tree = pickle.load(open('models/mlp_binary_d_dmax', 'rb'))

animals = df_new['animal'].unique()

res = []
xs = []
for animal in animals:
    res_anim = []
    df_a = df_new.loc[df_new["animal"] == animal]
    times = df_a['t'].unique()
    for time in times:
        df_t = df_a.loc[df_a["t"] == time]
        df_f = df_t.drop(columns=['animal', 't'])  # dropped
        X = StandardScaler().fit_transform(df_f)
        pred_tree = tree.predict(X)

        prob_t = np.count_nonzero(np.array(pred_tree) == 2) / len(pred_tree)
        prob_ne = (len(np.array(pred_tree)) - np.count_nonzero(np.array(pred_tree))) / len(pred_tree)
        prob_ent = np.count_nonzero(np.array(pred_tree) == 1) / len(pred_tree)

        if prob_t > 0.4:
            res_anim.append(2)
        elif prob_ent >= 0.6:
            res_anim.append(1)
        #elif prob_ne > 0.5:
           # res_anim.append(0)
        else:
            res_anim.append(3)

    res.append(res_anim)
    xs.append(times)


plt.xlabel('Time', fontsize=10)
plt.ylabel('Classification', fontsize=10)
plt.title('Stability of Classification', fontsize=15)
for i in range(20):
    plt.plot(xs[i], res[i], label="{}".format(animals[i]))
plt.legend(loc='best')
plt.show()
