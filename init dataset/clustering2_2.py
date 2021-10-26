# classification (train/validation)

#%%


from pylab import *
import pandas
df=pandas.read_csv('train_set1.csv')


#%%
df1=df
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))].dropna()

y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['animal','bf','group'])

sel=(0, 6, 7, 8, 10, 11)
df1=df1.iloc[:,array(sel)]

#var=list(df1)
#df1=df
#df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]
#var.append('group')
#df1.loc[:,var].dropna()

X=df1.values
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df1.values)


from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

#%%

from sklearn.model_selection import train_test_split
from numpy import random

accuracy_all=[]
for rr in random.randint(0,1000,5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=rr)
    
    # Learn to predict each class against the other
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import svm
    
    from sklearn.metrics import confusion_matrix
    #from sklearn.linear_model import LogisticRegression as algo
    #from sklearn.neighbors import KNeighborsClassifier as algo
    #from sklearn.ensemble import RandomForestClassifier as algo
    #from sklearn.tree import DecisionTreeClassifier as algo
    from sklearn.ensemble import AdaBoostClassifier as algo
    #from sklearn.naive_bayes import GaussianNB as algo
    
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.metrics import roc_curve, auc

    classifiers = [
        KNeighborsClassifier(3), \
        SVC(kernel="linear", C=0.025,probability=True),\
        SVC(gamma=2, C=1,probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),\
        DecisionTreeClassifier(max_depth=5),\
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),\
        MLPClassifier(alpha=1),\
        AdaBoostClassifier(),\
        GaussianNB(),\
        QuadraticDiscriminantAnalysis()]
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",\
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",\
             "Naive Bayes", "QDA"]
    
    for cl in range(len(names)):
        
        classifier = OneVsRestClassifier(classifiers[cl])
        #classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
        #y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        
        
        predictions=classifier.fit(X_train, y_train).predict(X_test)
        
        from sklearn.metrics import accuracy_score
        ac = accuracy_score(y_test,predictions)
        
        y_score = classifier.predict_proba(X_test)
        
        
        #%
        
        cm = confusion_matrix(y_test.dot([0,1,2]),predictions.dot([0,1,2]))
        

        # Compute ROC curve and ROC area for each class
        roc_auc = dict()
        fpr=dict()
        tpr=dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        accuracy_all.append({'r_state':rr,'classifier':names[cl],'score_ent':cm[0,0]/sum(cm,axis=0)[0], \
                             'score_et':cm[1,1]/sum(cm,axis=0)[1],\
                             'score_n':cm[2,2]/sum(cm,axis=0)[2],'auc_ent':roc_auc[0],'auc_et':roc_auc[1],'auc_n':roc_auc[2]})
    #    
    #    # Compute micro-average ROC curve and ROC area
    #    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #    
        fig=figure(figsize=(3.5,3))
        ax1=subplot(111)
        for i,c in zip(range(3),['b','r','k']):
            ax1.plot(fpr[i], tpr[i], color=c,label='%.2g'%(roc_auc[i]))
        plot([0, 1], [0, 1], color='k', linestyle=':')
        xlabel('FP')
        ylabel('TP')
        legend(loc=4)
        
#        left, bottom, width, height = [0.65, 0.3, 0.22, 0.22]
#        ax2 = fig.add_axes([left, bottom, width, height])
#        h2 = histogram2d(y_test.dot([0,1,2]),predictions.dot([0,1,2]),linspace(0,2,4))
#        ax2=pcolor(h2[1],h2[2],h2[0])
#        for lx,ly,lz in zip(list((array([0,1,2,0,1,2,0,1,2])*2+1)/3),list((array([0,0,0,1,1,1,2,2,2])*2+1)/3),h2[0].ravel()):
#            text(lx,ly,'%d'%(lz),ha='center',va='center',color='grey')
#        xticks((array([0,1,2,0,1,2,0,1,2])*2+1)/3,['ENT','ET','N'])
#        yticks((array([0,1,2,0,1,2,0,1,2])*2+1)/3,['ENT','ET','N'])
#        xlabel('Actual')
#        ylabel('Predicted')
#        xlim(0,2)
#        ylim(0,2)
        
        ax1.set_title(names[cl])
#        savefig(r'Z:/Calvin/Analysis/sync_reanalysis/drawing/classification'+str(cl)+'.pdf',bbox_inches='tight')
        close()
    
classcompare = pandas.DataFrame.from_records(accuracy_all)
print(classcompare)

#%% 
classcomp = classcompare.groupby('classifier').mean().drop('r_state',axis=1).iloc[:,:3]
classcomp=classcomp.sort_values(by='auc_ent',ascending=False)

figure(figsize=(10,3))
for i,cl in enumerate(classcomp.index):
    bar(i+arange(3)/4-1/4,classcomp.loc[cl,:].iloc[:3],1/4.5,color=['b','r','grey'])
xticks(range(i+1),classcomp.index.values,rotation=45,ha='right')
xlim([-0.5,i+0.5])
plot([-0.5,i+0.5],[0.5,0.5],'k:')
ylim(0,1)
ylabel('Accuracy')
savefig(r'Z:/Calvin/Analysis/sync_reanalysis/drawing/accuracy_compare.pdf',bbox_inches='tight')

#%%
from numpy import random
denom=5
figure(figsize=(7,3.5))
for rr in range(60):
    P={}
    for n_unit in range(60):
        prob_mat=ones((n_unit,denom))
        for i in range(n_unit):
            prob_mat[i,random.randint(denom)]=0
        P[n_unit]=sum(prob_mat[:,0])/n_unit
    plot(P.keys(),P.values(),'.')

plot([0,60],[1/denom,1/denom],'k:')
plot([0,60],[1-1/denom,1-1/denom],'k:')
ylim(-0.1,1.1)
xlabel('N unit')
ylabel('P')
savefig(r'Z:/Calvin/Analysis/sync_reanalysis/drawing/simulation1.pdf',bbox_inches='tight')


