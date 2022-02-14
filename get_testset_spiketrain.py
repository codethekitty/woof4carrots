from pylab import *
import sys,glob,os,pandas

#%%
flist=glob.glob('CW_GP_*')

for f in flist:
    dataset=[]
    tlist=glob.glob(os.path.join(f,'spike_detect','rec*'))
    t0=load(os.path.join(f,'FRA.npy'),allow_pickle=True)[1]['start']
    for t in tlist:
        sp=load(t,allow_pickle=True)[0]
        for k,v in sp.items():
            t_norm=int(t[-8:-6])*60+int(t[-6:-4])-t0
            dataset.append({'f':int(f[6:]),'t':t_norm,'ch':k,'ts':v})
        print(f,t)
    save('test_set_220211/spiketrain_'+f[6:]+'.npy',dataset)
    print(sys.getsizeof(dataset))
#%%
d=pandas.DataFrame.from_dict(dataset)

