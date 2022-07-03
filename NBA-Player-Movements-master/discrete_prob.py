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