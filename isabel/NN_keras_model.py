from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas
from sklearn.model_selection import train_test_split

df=pandas.read_csv('train_set1.csv')

df1 = df
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]
df1 = df1[~df1.group.str.contains("NE")]
remove1=df1.isna().any(axis=1)
df1=df1.dropna()
y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]


df1=df1.drop(columns=['animal','bf','group', 'bfr'])
X = StandardScaler().fit_transform(df1.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# first neural network with keras tutorial

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
np.random.seed(9)

model = Sequential()

# Layer 1
model.add(Dense(units=8, activation='sigmoid', input_dim=11))
# Layer 2
model.add(Dense(units=4, activation='sigmoid'))
# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)


model.fit(X_train, y_train, epochs=150, verbose=False, batch_size = 10)

print(model.predict(X_test))