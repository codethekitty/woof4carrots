import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('train_set1.csv')
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group', 'avg_ibi', 'avg_spikes_burst',
                              'max_spikes_burst', 'bfr', 'p_bursting_spikes',
                              'p_bursting_time', 'sfr'])  # dropped

# , 'bf', 'sync_n', 'max_sync_bf_dist', 'max_sync_coef', 'mean_sync_coef', 'mean_sync_bf_dist' not dropped

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()

X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y[~remove])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

tree = ExtraTreesClassifier(criterion='gini', bootstrap=True, class_weight='balanced_subsample', n_estimators=100)
svm_rbf = SVC(class_weight='balanced', C=1000)
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

'''
disp = ConfusionMatrixDisplay.from_estimator(
        svm_model_rbf,
        X_test,
        y_test,
        display_labels=['ENT', 'ET'],
        cmap=plt.cm.Blues,
    )

print(disp.confusion_matrix)
plt.show()
'''
