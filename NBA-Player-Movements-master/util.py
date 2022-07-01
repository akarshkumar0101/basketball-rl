import torch
import numpy as np
import matplotlib.pyplot as plt

def fourier_pos(tmin, tmax, t=None, d=512, do_viz=False):
    if t is None:
        t = torch.linspace(tmin, tmax, 1024)
    i = torch.arange(d)
    i_even = i[::2]
    i_odd = i[1::2]
    
    embed = torch.zeros(len(t), d)
    
    wl0 = 2*np.pi *     1 * (tmax-tmin) / 1024
    wl1 = 2*np.pi * 10000 * (tmax-tmin) / 1024
    wavelength = wl0 * (wl1/wl0)**(i_even/d)
    freq = 1./wavelength
    
    a = t[:, None] * 2*np.pi/wavelength
    sin = embed[:, i_even] = a.sin()
    cos = embed[:, i_odd] = a.cos()
    
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