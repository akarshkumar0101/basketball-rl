import numpy as np

class DiscretizeContinuousSpace():
    def __init__(self, start_point, end_point, n_bins):
        self.p1, self.p2 = start_point, end_point
        self.n_bins = n_bins
        print(self.p1, self.p2, self.n_bins)

    def vec2bin(self, v):
        v = (v-self.p1)/(self.p2-self.p1)*self.n_bins
        idx = v.astype(int).clip(0, self.n_bins-1)
        idx = np.moveaxis(idx, -1, 0)
        idx = np.ravel_multi_index(tuple(idx), self.n_bins)
        return idx
    
    def bin2vec(self, idx):
        idx = np.unravel_index(idx, self.n_bins)
        idx = np.moveaxis(np.stack(idx), 0, -1)
        v = (idx+.5)/self.n_bins*(self.p2-self.p1)+self.p1
        return v
    
    
# to visualize:

# dcs = DiscretizeContinuousSpace(np.zeros(2)-1.5, np.zeros(2)+1.5, np.ones(2, dtype=int)*10)
# plt.figure(figsize=(20, 20))
# c = np.random.rand(100, 3)
# x = np.random.randn(10000, 2)
# idx = dcs.vec2bin(x)
# plt.scatter(*x.T, c=c[idx], s=10) # make sure vec2bin is working

# idx = np.arange(0, 100)
# v = dcs.bin2vec(idx)
# c = np.random.rand(100, 3)
# plt.scatter(*v.T, c=c[idx], s=1000, marker='x') # make sure bin2vec is working

# plt.gca().axhline(0)
# plt.gca().axvline(0)