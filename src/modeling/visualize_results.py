import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

mu, sig, o , m = torch.load('cvae.predictions.pt')

par_mu = mu[0].data.numpy()
par_sig = sig[0].data.numpy()
obs = o[0].numpy()
msk = m[0].numpy()
s = np.random.lognormal(par_mu[2,5], par_sig[2,5], 1000)

plt.figure()
figdist = sns.distplot(s, kde=False, color="b")
figdist.set_yscale('log')
figdist.axvline(obs[2,5], color='k', linestyle='--')
plt.savefig('figdist.png')

rr = None
mm = None
for i in range(10):
    par_mu = mu[i].data.numpy()
    par_sig = sig[i].data.numpy()
    obs = o[i].numpy()
    msk = m[i].numpy()
    r = 2 * (1 - scipy.stats.norm(par_mu, par_sig).cdf(np.abs(np.log(obs))))
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


# For now this just compares two random individuals
rr = None
for i in range(10):
    par_mu0 = mu[i].data.numpy()
    par_sig0 = sig[i].data.numpy()
    par_mu1 = mu[i+100].data.numpy()
    par_sig1 = sig[i+100].data.numpy()
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
figsummary = sns.heatmap(df,  cmap="YlGnBu", yticklabels=False)
plt.xticks(rotation=45)
plt.savefig('figcomparison.png')
