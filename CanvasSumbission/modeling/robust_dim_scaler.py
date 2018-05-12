from copy import copy

class RobustDimScaler:
    def _getNonMissIndx(self, arr):
        return abs(arr - self.missing_val) > self.toler_eps

    def __init__(self, dataTrain, missing_val, qunatile_min, quantile_max, toler_eps=1e-5):
        if len(dataTrain.shape) != 2:
            raise Exception('This works only for 2d matrices')
        self.dim = dataTrain.shape[1]
        self.missing_val = missing_val
        self.toler_eps = toler_eps

        self.scalers = []


        for i in range(self.dim):
            indx = self._getNonMissIndx(dataTrain[:,i])
            if sum(indx) > 0:
                arr = dataTrain[indx,i]
                qmin, qmax = np.percentile(arr, (qunatile_min, quantile_max))
                self.scalers.append( (qmin, 1.0/(qmax - qmin)) )
            else:
                self.scalers.append((0, 1.0 ))

    def transform(self, data):
        if len(data.shape) != 2:
            raise Exception('This works only for 2d matrices')
        if data.shape[1] != self.dim:
            raise Exception('Missmtaching dimension between train (%d) and test (%d) data' %
                            (self.dim, data.shape[1]))

        for i in range(self.dim):
            indx = self._getNonMissIndx(data[:,i])
            arr = copy(data[indx, i])

            qmin, scale = self.scalers[i]
            arr = np.maximum((arr - qmin) * scale, self.toler_eps)

            data[indx, i] = arr

        return data



import numpy as np

np.random.seed(0)

nr = 20
nc = 8

data = np.full((nr, nc), 0.0).astype(np.float32)

for i in range(nr):
    data[i,:] = np.random.uniform(0, 100, nc)

for k in range(int(nr*nc / 4)):
    r = np.random.randint(0, nr)
    c = np.random.randint(0, nc)
    data[r,c] = -1


#print(data)

#print('==============')

scalar = RobustDimScaler(data, -1, 5, 95)

scalar.transform(data)

#print(data)

