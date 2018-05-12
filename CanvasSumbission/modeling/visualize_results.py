import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

#mu, sig, o , m = torch.load('cvae.predictions.pt.baseline.old',map_location=lambda storage, loc: storage)
mu, sig, o , lods, m = torch.load('predictions1/cvae.predictions.pt.overall',map_location=lambda storage, loc: storage)
mu0, sig0, o0 , lods0, m0 = torch.load('predictions1/cvae.predictions.pt.baseline',map_location=lambda storage, loc: storage)
mu1, sig1, o1 , lods1, m1 = torch.load('predictions1/cvae.predictions.pt.counterfactual',map_location=lambda storage, loc: storage)

#idx = np.random.choice(list(range(len(mu))),1,replace=False)
idxTrain = list(range(0,3896))
idxValid = list(range(3896,3896+400))
idxTest = list(range(3896+400,3896+800))

#par_mu = mu[500].data.numpy()
#par_sig = sig[500].data.numpy()
#obs = o[500].numpy()
#msk = m[500].numpy()
#print(par_mu[2,5], par_sig[2,5],obs[2,5])
#s = np.random.lognormal(par_mu[2,5], par_sig[2,5], 1000)
#plt.figure()
#figdist = sns.distplot(s, kde=False, color="b")
#figdist.set_yscale('log')
#figdist.axvline(obs[2,5], color='k', linestyle='--')
#plt.savefig('figdist.png')

rr = None
mm = None
for i in idxTest:
    par_mu = mu[i].data.numpy()
    par_sig = sig[i].data.numpy()
    obs = o[i].numpy()
    msk = m[i].numpy()
    #print(par_mu,par_sig,obs,msk)
    r = np.minimum(1,2 * (1 - scipy.stats.norm(par_mu, par_sig).cdf(np.abs(np.log(obs)))))
    #print(r)
    mf = 1-msk
    if rr is None:
        rr = r
        mm = mf
    else:
        rr = np.concatenate((rr,r),0)
        mm = np.concatenate((mm, mf), 0)

df = pd.DataFrame(rr, columns=["URXUHG","URXUCD","URXUPB","LBXBGM","LBXBCD","LBXBPB","LBXTHG","URXUAS"])
sns.set(font_scale=1.5)
plt.figure(figsize=(10,10))
figsummary = sns.heatmap(df,  cmap="YlGnBu", mask = mm,  yticklabels=False)
plt.xticks(rotation=45)
plt.savefig('figsummary.png')

rr = None
mm = None
rm = None
for i in idxTest:
    par_mu = mu[i].data.numpy()
    par_sig = sig[i].data.numpy()
    obs = o[i].numpy()
    msk = m[i].numpy()
    lod = lods[i].numpy()
    r = np.minimum(1,2 * (1 - scipy.stats.norm(par_mu, par_sig).cdf(np.abs(np.log(obs)))))
    #print(r)
    rmi = r * msk
    if rr is None:
        rr = r
        mm = msk
        rm = rmi
    else:
        rr = np.concatenate((rr,r),0)
        mm = np.concatenate((mm, msk), 0)
        rm = np.concatenate((rm, rmi), 0)

df = pd.DataFrame(rm, columns=["URXUHG","URXUCD","URXUPB","LBXBGM","LBXBCD","LBXBPB","LBXTHG","URXUAS"])
print(df.sum())
df2 = pd.DataFrame(mm, columns=["URXUHG","URXUCD","URXUPB","LBXBGM","LBXBCD","LBXBPB","LBXTHG","URXUAS"])
print(df2.sum())
print(df.sum()/df2.sum())

# For now this just compares two random individuals

rr = None
for i in range(len(mu0)):
    par_mu0 = mu0[i].data.numpy()
    par_sig0 = sig0[i].data.numpy()
    par_mu1 = mu1[i].data.numpy()
    par_sig1 = sig1[i].data.numpy()
    res = np.zeros(par_mu0.shape)
    for r in range(par_mu0.shape[0]):
        for c in range(par_mu0.shape[1]):
            s0 = np.random.lognormal(par_mu0[r,c], par_sig0[r,c], 1000)
            s1 = np.random.lognormal(par_mu1[r,c], par_sig1[r,c], 1000)
            out = np.sum((s1 < s0))
            res[r,c] = out /1000.0
    if rr is None:
        rr = res
    else:
        rr = np.concatenate((rr,res),0)

df = pd.DataFrame(rr, columns=["URXUHG","URXUCD","URXUPB","LBXBGM","LBXBCD","LBXBPB","LBXTHG","URXUAS"])
sns.set(font_scale=1.5)
plt.figure(figsize=(10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#figsummary = sns.heatmap(df,  cmap="YlGnBu", yticklabels=False)
figsummary = sns.heatmap(df,  cmap=cmap, center=0.5, yticklabels=False)
plt.xticks(rotation=45)
plt.savefig('figcomparison.png')

rr = None
for i in range(len(mu0)):
    par_mu0 = mu0[i].data.numpy()
    par_sig0 = sig0[i].data.numpy()
    par_mu1 = mu1[i].data.numpy()
    par_sig1 = sig1[i].data.numpy()
    res = np.zeros(par_mu0.shape)
    for r in range(par_mu0.shape[0]):
        for c in range(par_mu0.shape[1]):
            s0 = np.random.lognormal(par_mu0[r,c], par_sig0[r,c], 1000)
            s1 = np.random.lognormal(par_mu1[r,c], par_sig1[r,c], 1000)
            out = np.sum((s1 < s0))
            res[r,c] = out /1000.0
    if rr is None:
        rr = res
    else:
        rr = np.concatenate((rr,res),0)

df3 = pd.DataFrame(rr, columns=["URXUHG","URXUCD","URXUPB","LBXBGM","LBXBCD","LBXBPB","LBXTHG","URXUAS"])
print(df3.mean())
