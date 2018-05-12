import numpy as np
#pred = np.load('data/npy/predictorSample.npy')
pred = np.load('../Data/data_adult_2007-2014.npy')
idx = list(range(pred.shape[0]))
idxSel = []

for i in idx:
    suppl = 0
    for s in range(14083,32612):
        if pred[i,s]==1:
            suppl = suppl +1
    #print(suppl)
    if suppl > 0 :
        idxSel.append(i)

print(len(idx))
print(len(idxSel))
#print(idxSel)
#np.save('data/npy/supplConsumers.npy',idxSel)
np.save('../Data/supplConsumers_adult_2007-2014.npy',idxSel)