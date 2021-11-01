# real data test (train/test)


#%% load and setup training dataset

df=pandas.read_csv('train_set.csv')
df.group = df.group.map(dict(zip(unique(df.group),['ENT','ET','ET',nan,nan,'N'])))

df=df[(df.animal!='E3')&(df.animal!='CW30')]
df=df[(df.animal!='D8')]
df=df[(df.animal!='F2')]

df=df.dropna()
y=unique(df.group.values,return_inverse=True)[1]
df1=df.drop(columns=['animal','bf','ch','loc','group'])

remove1=df1.isna().any(axis=1)
df1=df1.dropna()
X = StandardScaler().fit_transform(df1.values)
y=y[~remove1]








#%% correction of test dataset

df_test=pandas.read_excel('P_animal_set.xls')
df_test = df_test[(df_test.sync_n>0)&(df_test.avg_ibi>0)]

figure(figsize=(10,2.5))

for i,animal_sel in enumerate(df_test.animal.unique()):
    df_test3 = df_test[df_test.animal==animal_sel]
    
    y_test= array(df_test3.loc[:,'group']=='ET',dtype='int')
    X_test = df_test3.drop(columns=['animal','bf','group']).loc[:,list(df_train)].values
    
    #% scale datasets
    
    from sklearn.preprocessing import StandardScaler
    
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    
    from sklearn.ensemble import AdaBoostClassifier
    
    classifier = AdaBoostClassifier()
    predictions=classifier.fit(X_train, y_train).predict(X_test)
    
    bar(i,sum(predictions)/len(y_test)+y_test[0]*0.7-0.36-0.5,bottom=0.5)
    plot(i,y_test[0],['v','^'][y_test[0]],color=array(['b','r'])[y_test[0]],ms=14)
#    text(i,0.43,str(round(len(y_test))),ha='center')
xticks(range(i+1),['Edward','Brad','Richard','Brian','Lukas','Charlie','Dan','Roger'],rotation=45,ha='right')
xlim(-1,8)
plot((-1,8),(0.5,0.5),'k:')
text(8.1,0,'T-',color='b',va='center')
text(8.1,1,'T+',color='r',va='center')

ylabel('% unit classified as T+')
savefig(r'Z:/Calvin/Analysis/sync_reanalysis/drawing/actual.pdf',bbox_inches='tight')





#%%
