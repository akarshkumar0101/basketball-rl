import torch
import numpy as np
import matplotlib.pyplot as plt

import copy

def fourier_pos(tmin, tmax, t=None, d=512, do_viz=False):
    if t is None:
        t = torch.linspace(tmin, tmax, 1024)
    i = torch.arange(d).to(t.device)
    i_even = i[::2]
    i_odd = i[1::2]
    
    embed = torch.zeros(*t.shape, d).to(t.device)
    
    wl0 = 2*np.pi *     1 * (tmax-tmin) / 1024
    wl1 = 2*np.pi * 10000 * (tmax-tmin) / 1024
    wavelength = wl0 * (wl1/wl0)**(i_even/d)
    freq = 1./wavelength
    
    a = t[..., None] * 2*np.pi/wavelength
    sin = embed[..., i_even] = a.sin()
    cos = embed[..., i_odd] = a.cos()
    
    if do_viz:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.pcolormesh(i_even.numpy(), t.numpy(), sin.numpy()); plt.colorbar()
        plt.ylim(tmax, tmin)

        plt.subplot(132)
        plt.pcolormesh(i_odd.numpy(), t.numpy(), cos.numpy()); plt.colorbar()
        plt.ylim(tmax, tmin)

        plt.subplot(133)
        plt.pcolormesh(i.numpy(), t.numpy(), embed.numpy()); plt.colorbar()
        plt.ylim(tmax, tmin)
        plt.show()
    return embed

def sliding_window(n_window, n_total):
    idxs = []
    for i in range(n_window):
        idxs.append(torch.arange(n_total-n_window+1)+i)
    return torch.stack(idxs).T

def print_data_dict(data):
    for key, value in data.items():
        print(f'{key}: {value.shape}', end=' | ')
    print(); print()
    
def dict_list2list_dict(data):
    ans = []
    listlen = len(list(data.values())[0])
    for i in range(listlen):
        ans.append({key: value[i] for key, value in data.items()})
    return ans

def list_dict2dict_list(data):
    ans = {}
    
    for key in data[0].keys():
        ans[key] = [di[key] for di in data]
    
    return ans