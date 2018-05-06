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
