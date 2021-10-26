# parameter search

#%%
from pylab import*
import pandas

df=pandas.read_csv('train_set1.csv')

from sklearn.preprocessing import StandardScaler
df1=df
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]

y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['animal','bf','group'])

remove1=df1.isna().any(axis=1)
df1=df1.dropna()
X = StandardScaler().fit_transform(df1.values)
y=y[~remove1]

#%% k-fold split 
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

for n in arange(2,12,2):
    kf = KFold(n_splits=n)
    ac=[]
    for train_index, test_index in kf.split(X):
    #    print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        predictions=classifier.fit(X_train, y_train).predict(X_test)
        ac.append(accuracy_score(y_test,predictions))
    plot([n]*len(ac),ac,'o')
    errorbar(n+0.2,mean(ac),std(ac),marker='s',ls='-')

#%% combinations of parameters

def n_combination(n,r):
    c = math.factorial(n)/math.factorial(n-r)/math.factorial(r)
    return c
ctotal=0
for i in arange(2,12):
    c=n_combination(12,i)
    plot(i,c,'o')
    text(i,c,str(c))
    ctotal+=c
print(ctotal)
    
    
#%%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations 
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import confusion_matrix
from numpy import random

X_train_all=[]
y_train_all=[]

classifier=LDA(n_components=2)
result_all=[]
for rr in random.randint(0,1000,10):
    for nn in arange(2,12,1):
        c = combinations(range(12),nn)
        for i in c:
            X_sub = X[:,i]
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=.2,random_state=rr)
            
            # 2d only
            X_train = classifier.fit_transform(X_train,y_train)
            X_test = classifier.transform(X_test)
            classifier.fit(X_train, y_train)
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            # all d
#            classifier.fit(X_train, y_train)
            
            predictions=classifier.predict(X_test)
            ac = accuracy_score(y_test,predictions)
            cm = confusion_matrix(y_test,predictions)
            
            with errstate(divide='ignore', invalid='ignore'):
                d={'rstate':rr,'n_combo':nn,'vars_index':i,'score':ac, \
                   'score_ent':cm[0,0]/sum(cm,axis=0)[0], \
                   'score_et':cm[1,1]/sum(cm,axis=0)[1], \
                   'score_n':cm[2,2]/sum(cm,axis=0)[2]}
            result_all.append(d)
            print(d)
            
print('done')


        

#%% accuracy vs n param
cpalette=rcParams['axes.prop_cycle'].by_key()['color']

figure(figsize=(10,3),dpi=150)
import matplotlib.gridspec as gridspec

lda_result = pandas.DataFrame.from_records(result_all).dropna()
g = lda_result.groupby(['vars_index']).mean().drop(columns='rstate')

gs = gridspec.GridSpec(1,len(g.n_combo.unique()))

for i,n in enumerate(sort(g.n_combo.unique())):
    subplot(gs[i])
    dent = g[g.n_combo==n].score_ent.sort_values().values
    det = g[g.n_combo==n].score_et.sort_values().values
    dn = g[g.n_combo==n].score_n.sort_values().values
    plot(dent,range(len(dent)),color='b',label='ENT')
    plot(det,range(len(det)),color='r',label='ET')
    plot(dn,range(len(dn)),color='k',label='N')
    ylim(0,800)
    xlim(0,1)
    yticks([])
    xticks([0,1])
    title('%d'%(n),fontsize=10)
subplot(gs[5])
xlabel('Classification accuracy')
text(-0.5,900,'N parameters',fontsize=10)
subplot(gs[0])
ylabel('N combos')


#%%
figure(figsize=(7.5,4.5))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 1,height_ratios=(2,1))

g2 = g.dropna().iloc[:,2:]

cvcomp = g2.std(axis=1)/g2.mean(axis=1)
maxcomp = sum(g2,axis=1)

subplot(gs[0])

plot(sort(maxcomp),label='Summed accuracy')
plot(cvcomp[argsort(maxcomp)].values,label='CV')
legend(loc=2)
xlim(0,len(maxcomp))
xticks([])
subplot(gs[1])
lx = array([len(x) for x in maxcomp.sort_values().index])
plot(lx,c=co[2])
ylim(0,12)
xlim(0,len(maxcomp))
#plot(g2.values[argsort(maxcomp)][:,0],color='b',label='ENT')
#plot(g2.values[argsort(maxcomp)][:,1],color='r',label='ET')
#plot(g2.values[argsort(maxcomp)][:,2],color='#555555',label='N')
ylabel('N parameters')
xlabel('Combo index')
savefig(r'Z:\Calvin\Analysis\sync_reanalysis\drawing\accuracy1.pdf',bbox_inches='tight')

#%%

plot(maxcomp,cvcomp,'go',ms=2)
xlabel('Max accuracy')
ylabel('CV')

tight_layout()

#savefig(r'Z:\Calvin\Analysis\sync_reanalysis\drawing\accuracy1.pdf',bbox_inches='tight')

#%%
#for combo in maxcomp[maxcomp>2.8].index:
#    lp = array(list(df1))[array(combo)]
sel=(10, 11, 6, 7, 0, 8)
df2=df1.iloc[:,array(sel)]
df2['group']=array(['ENT','ET','N'])[y]

figure(figsize=(18,3))
import seaborn as sns
gs = gridspec.GridSpec(1,size(df2,axis=1))
for i,col in enumerate(array(list(df2))[:-1]):
    subplot(gs[i])          
    sns.boxplot(x='group',y=col,data=df2,palette=['r','b','grey'],linewidth=0.5,fliersize=2,whis=1.5)   
    xticks([])
    xlabel('')
tight_layout()

savefig(r'Z:\Calvin\Analysis\sync_reanalysis\drawing\params.pdf',bbox_inches='tight')



