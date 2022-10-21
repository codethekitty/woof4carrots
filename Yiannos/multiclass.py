import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn import tree

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["group"] != "ET_A"]  # remove ET_A data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET": 2}, inplace=True)
df["group"].replace({"ET_E": 2}, inplace=True)  # mix data-points with ET
df["group"].replace({"ENT": 1}, inplace=True)
df["group"].replace({"NE": 0}, inplace=True)

df_new = df
df_new = df_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv',  'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'br', 'sfr', 'r', 'r_max'])  # dropped

#  'd_max', 'bf_deviation',  'bf', 'd', 'sync_n',  not dropped

df_new = df_new.dropna()
y = df_new.loc[:, "group"]
df_new = df_new.drop(columns=['group'])
X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y)

tree_ovo = OneVsOneClassifier(ExtraTreesClassifier(bootstrap=True, class_weight='balanced_subsample', max_depth=7, min_samples_leaf=10))
knn_ovo = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1, weights='distance'))
svm_ovo = OneVsOneClassifier(SVC(class_weight='balanced', C=100))


# Tree picture
'''
fig = plt.figure(figsize=(200, 200))
_ = tree.plot_tree(tree_ovo.estimators_[0][0], feature_names=df_new.columns, class_names=['NE', 'ENT', 'ET'], filled=True)
plt.savefig('tree_4+syncn_depth7_min10_50')
'''


# Confusion Matrix
'''
disp = ConfusionMatrixDisplay.from_estimator(
        tree_ovr,
        X_test,
        y_test,
        display_labels=['NE', 'ENT', 'ET'],
        cmap=plt.cm.Blues,
    )

print(disp.confusion_matrix)
plt.show()
'''

# Save Models
'''
pickle.dump(tree_ovo, open('tree_ovo_rmax', 'wb'))
pickle.dump(knn_ovo, open('knn_ovo_rmax', 'wb'))
pickle.dump(svm_ovo, open('svm_ovo_rmax', 'wb'))
'''

# Model Comparison
'''
result = [[], [], [], []]
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    tree_ovo.fit(X_train, np.array(y_train).ravel())
    svm_ovo.fit(X_train, np.array(y_train).ravel())
    knn_ovo.fit(X_train, np.array(y_train).ravel())

    result[0].append(tree_ovo.score(X_test, y_test))
    result[1].append(svm_ovo.score(X_test, y_test))
    result[2].append(knn_ovo.score(X_test, y_test))

plt.xlabel('Run', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Model Comparison', fontsize=15)
x = [i for i in range(100)]
plt.plot(x, result[0], label='Tree_ovo')
plt.plot(x, result[1], label='SVM_ovo')
plt.plot(x, result[2], label='KNN_ovo')
plt.legend(loc='best')

plt.show()
'''

'''
# Feature comparison

def make_dataset(drop_features):
    df_new = df
    df_new = df_new.drop(columns=drop_features)  # dropped
    df_new = df_new.dropna()
    y = df_new.loc[:, "group"]
    df_new = df_new.drop(columns=['group'])
    X = StandardScaler().fit_transform(df_new)
    y = pd.DataFrame(y)
    return X, y


def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree_ovr = OneVsOneClassifier(ExtraTreesClassifier(bootstrap=True, class_weight='balanced_subsample'))
    knn_ovr = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1, weights='distance'))
    svm_ovr = OneVsOneClassifier(SVC(class_weight='balanced', C=10000))

    tree_ovr.fit(X_train, np.array(y_train).ravel())

    return tree_ovr.score(X_test, y_test)


features = ['animal', 'loc', 'ch', 'isi_cv', 'sfr',
            'br', 'bdur_max', 'bdur', 'nspikes_burst_max', 'nspikes_burst',
            'p_bursting_time', 'p_bursting_spike', 'ibi_cv',
            'bf_deviation', 'bf', 'sync_n', 'r_max', 'r', 'd_max', 'd']

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'br')]
X_max, y_max = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'r_max')]
X_bdur, y_bdur = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'ibi_cv')]
X_r, y_r = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation')]
X_base, y_base = make_dataset(ft)

result = [[], [], [], [], []]
for j in range(20):
    res = [[], [], [], [], []]

    for i in range(50):
        res[0].append(run_model(X_max, y_max))
        res[1].append(run_model(X_bdur, y_bdur))
        res[2].append(run_model(X_r, y_r))
        res[3].append(run_model(X_base, y_base))

    result[0].append(sum(res[0]) / len(res[0]))
    result[1].append(sum(res[1]) / len(res[1]))
    result[2].append(sum(res[2]) / len(res[2]))
    result[3].append(sum(res[3]) / len(res[3]))


plt.xlabel('Run', fontsize=10)
plt.ylabel('Average Accuracy', fontsize=10)
plt.title('Feature Comparison', fontsize=15)
x = [i for i in range(20)]
plt.plot(x, result[0], label='base + br')
plt.plot(x, result[1], label='base + r_max')
plt.plot(x, result[2], label='base + ibi_cv')
plt.plot(x, result[3], label='base')
plt.legend(loc='best')

plt.show()
'''

