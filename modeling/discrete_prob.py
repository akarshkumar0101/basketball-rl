import numpy as np

import torch

# def unravel_index(index, shape):
#     """
#     https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
#     """
#     out = []
#     for dim in reversed(shape):
#         out.append(index % dim)
#         index = index // dim
#     return tuple(reversed(out))

class DiscretizeContinuousSpace():
    def __init__(self, start_point, end_point, n_bins):
        """
        start_point.shape is (d, ) of type float indicating lower bound.
        end_point.shape is (d, ) of type float indicating upper bound.
        n_bins.shape is (d, ) of type int indicated number of bins in each dimension.
        """
        self.p1, self.p2 = np.array(start_point), np.array(end_point)
        self.n_bins = np.array(n_bins)
        # print(self.p1, self.p2, self.n_bins)

    def vec2bin(self, v):
        """
        v.shape is (..., d) where n_bins.shape is (d, )
        output.shape is (...,) indicating the flattened bin idxs.
        """
        v = (v-self.p1)/(self.p2-self.p1)*self.n_bins # normalize to scale 0-n_bins[i] in each dimension i
        idx = v.astype(int).clip(0, self.n_bins-1) # take int and clip to get bin idxs in each dimension
        idx = np.moveaxis(idx, -1, 0) # move the vector dimension to the front so I can tuple it
        idx = np.ravel_multi_index(tuple(idx), self.n_bins) # convert tuple of indices into flattened bins
        return idx
    
    def vec2bin_torch(self, v):
        return torch.from_numpy(self.vec2bin(v.numpy()))
    
    def bin2vec(self, idx):
        """
        idx.shape is (...,) indicating the flattened bin idxs.
        output.shape is (..., d) indicating the center vector of that bin idx
        """
        idx = np.unravel_index(idx, self.n_bins)
        idx = np.moveaxis(np.stack(idx), 0, -1)
        v = (idx+.5)/self.n_bins*(self.p2-self.p1)+self.p1
        return v
    
    def bin2vec_torch(self, idx):
        return torch.from_numpy(self.bin2vec(idx.numpy()))
    
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