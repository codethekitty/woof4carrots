import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn import tree, metrics

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["group"] != "ET_A"]  # remove ET_A data-points
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET": 2}, inplace=True)
df["group"].replace({"ET_E": 2}, inplace=True)  # mix data-points with ET
df["group"].replace({"ENT": 1}, inplace=True)
#df["group"].replace({"NE": 0}, inplace=True)


df_new = df
df_new = df_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv',  'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'br', 'r', 'sfr', 'bf_deviation', 'sync_n', 'r_max'])  # dropped

#  'd_max', 'd', 'bf'   'group'  not dropped

df_new = df_new.dropna()
y = df_new.loc[:, "group"]
df_new = df_new.drop(columns=['group'])
X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y)

# changed to non multiclass to test something - change back to ovo if multiclass wanted
tree_ovo = ExtraTreesClassifier(class_weight='balanced_subsample', min_samples_leaf=20, n_estimators=500)
knn_ovo = KNeighborsClassifier(n_neighbors=40, weights='distance', p=1) # manhatan distance to minimize effect of outliers
svm_ovo = SVC(class_weight='balanced', C=0.1)
mlp = MLPClassifier(solver='lbfgs', max_iter=1500, alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(tree_ovo.fit(X_train, np.array(y_train).ravel()).score(X_test, y_test))
print(svm_ovo.fit(X_train, np.array(y_train).ravel()).score(X_test, y_test))
print(knn_ovo.fit(X_train, np.array(y_train).ravel()).score(X_test, y_test))
print(mlp.fit(X_train, np.array(y_train).ravel()).score(X_test, y_test))

pickle.dump(mlp, open('models/mlp_binary_d_dmax_bf', 'wb'))

# Tree picture
'''
fig = plt.figure(figsize=(200, 200))
_ = tree.plot_tree(tree_ovo.estimators_[50], feature_names=df_new.columns, class_names=['ENT', 'ET'], filled=True)
plt.savefig('tree_4+syncn+r_max_depth4')
'''


# Confusion Matrix
'''
disp = ConfusionMatrixDisplay.from_estimator(
        tree_ovo,
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
pickle.dump(tree_ovo, open('models/tree_ovo', 'wb'))
pickle.dump(knn_ovo, open('models/knn_ovo', 'wb'))
pickle.dump(svm_ovo, open('models/svm_ovo', 'wb'))
pickle.dump(mlp, open('models/mlp', 'wb'))
'''

# Model Comparison
'''
result = [[], [], [], []]
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree_ovo.fit(X_train, np.array(y_train).ravel())
    svm_ovo.fit(X_train, np.array(y_train).ravel())
    knn_ovo.fit(X_train, np.array(y_train).ravel())
    mlp.fit(X_train, np.array(y_train).ravel())

    result[0].append(tree_ovo.score(X_test, y_test))
    result[1].append(svm_ovo.score(X_test, y_test))
    result[2].append(knn_ovo.score(X_test, y_test))
    result[3].append(mlp.score(X_test, y_test))

plt.xlabel('Run', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Model Comparison', fontsize=15)
x = [i for i in range(50)]
plt.plot(x, result[0], label='Tree_ovo')
plt.plot(x, result[1], label='SVM_ovo')
plt.plot(x, result[2], label='KNN_ovo')
plt.plot(x, result[3], label='MLP')
plt.legend(loc='best')

plt.show()
'''

# Feature comparison

'''
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

    tree_ovr = OneVsOneClassifier(ExtraTreesClassifier(class_weight='balanced_subsample', min_samples_leaf=10, max_depth=13))
    tree_ovr.fit(X_train, np.array(y_train).ravel())

    return tree_ovr.score(X_test, y_test)


features = ['animal', 'loc', 'ch', 'isi_cv', 'sfr',
            'br', 'bdur_max', 'bdur', 'nspikes_burst_max', 'nspikes_burst',
            'p_bursting_time', 'p_bursting_spike', 'ibi_cv',
            'bf_deviation', 'bf', 'sync_n', 'r_max', 'r', 'd_max', 'd']

ft = [feat for feat in features if feat not in ('bf', 'd', 'd_max', 'bf_deviation', 'r_max', 'sfr')]
X_max, y_max = make_dataset(ft)

ft = [feat for feat in features if feat not in ('bf', 'd_max', 'd', 'r_max', 'bf_deviation', 'sync_n')]
X_bdur, y_bdur = make_dataset(ft)

ft = [feat for feat in features if feat not in ('bf', 'r_max', 'd_max', 'd', 'bf_deviation', 'sfr', 'sync_n')]
X_r, y_r = make_dataset(ft)

ft = [feat for feat in features if feat not in ('bf', 'd_max', 'd', 'bf_deviation', 'r_max')]
X_base, y_base = make_dataset(ft)

result = [[], [], [], [], []]
for j in range(10):
    res = [[], [], [], [], []]

    for i in range(10):
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
x = [i for i in range(10)]
plt.plot(x, result[0], label='base +sfr')
plt.plot(x, result[1], label='base +sync_n')
plt.plot(x, result[2], label='base +both')
plt.plot(x, result[3], label='base')
plt.legend(loc='best')

plt.show()
'''

