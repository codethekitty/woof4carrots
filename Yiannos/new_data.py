import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET_A": "ET"}, inplace=True)  # mix data-points with ET
df["group"].replace({"ET_E": "ET"}, inplace=True)  # mix data-points with ET
df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group', 'loc', 'ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n'])  # dropped

# , 'bf_deviation', 'bf', 'd_max', 'd'  not dropped

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()

X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y[~remove])

tree = ExtraTreesClassifier(criterion='gini', bootstrap=True, class_weight='balanced_subsample', n_estimators=100)
svm_rbf = SVC(class_weight='balanced', C=10000)
knn = KNeighborsClassifier(n_neighbors=1, weights='distance')

result = [[], [], [], []]
for i in range(200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    tree.fit(X_train, np.array(y_train).ravel())
    svm_rbf.fit(X_train, np.array(y_train).ravel())
    knn.fit(X_train, np.array(y_train).ravel())

    result[0].append(tree.score(X_test, y_test))
    result[1].append(svm_rbf.score(X_test, y_test))
    result[2].append(knn.score(X_test, y_test))

plt.xlabel('Run', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Model Comparison', fontsize=15)
x = [i for i in range(200)]
plt.plot(x, result[0], label='Tree')
plt.plot(x, result[1], label='SVM_rbf')
plt.plot(x, result[2], label='KNN')
plt.legend(loc='best')

plt.show()
