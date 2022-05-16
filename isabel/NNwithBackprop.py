#Neural network with backprop

from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas
from sklearn.model_selection import train_test_split

df=pandas.read_csv('train_set1.csv')

df1 = df
df1 = df1[~df1.group.str.contains("NE")]
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]

remove1=df1.isna().any(axis=1)
df1=df1.dropna()
y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]


df1=df1.drop(columns=['animal','bf','group', 'bfr'])
X = StandardScaler().fit_transform(df1.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


bias = np.array([0.0])

#basic NN

# =============================================================================
# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input      = x
#         self.weights1   = np.random.rand(self.input.shape[1],4) 
#         self.weights2   = np.random.rand(4,1)                 
#         self.y          = y
#         self.output     = np.zeros(y.shape)
# =============================================================================

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
        
def sigmoid_derivative(x):
    return x * (1.0 - x)

 #assume bias is 0       
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],307) 
        self.weights2   = np.random.rand(307,307)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer1
        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        

nn = NeuralNetwork(X_train, y_train)
for i in range(435):
        nn.feedforward()
        nn.backprop()
        
print(nn.output)

