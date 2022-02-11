from pylab import *
import sys,glob,os,pandas

#%%
flist=glob.glob('CW_GP_*')

for f in flist:
    dataset=[]
    tlist=glob.glob(os.path.join(f,'spike_detect','rec*'))
    for t in tlist:
        sp=load(t,allow_pickle=True)[0]
        for k,v in sp.items():
            dataset.append({'f':int(f[6:]),'t':int(t[-8:-4]),'ch':k,'ts':v})
        print(f,t)
    save('test_set_220211/spiketrain_'+f[6:]+'.npy',dataset)
    print(sys.getsizeof(dataset))
#%%
d=pandas.DataFrame.from_dict(dataset)

