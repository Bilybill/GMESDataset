#%%
import numpy as np
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":
    npz_path = '/home/wangyh/DATAFOLDER/samples/train-choas/crossed/AYL-00000.npz'
    data = np.load(npz_path)
    # print(list(data.keys())) 
    
    Details = data['formatS']
    MDetails = data['formatD']
    Trends = data['gtime'] # Dimensions: x * y * z
    
    vmin = 2000
    vmax = 6000
    Trends = (Trends - Trends.min()) / (Trends.max() - Trends.min())
    Trends = vmin + Trends * (vmax - vmin)

    vmin = -500
    vmax = 1000
    Details = (Details - Details.min()) / (Details.max() - Details.min())
    Details = vmin + Details * (vmax - vmin)
    
    vmin = -200
    vmax = 200
    MDetails = (MDetails - MDetails.min()) / (MDetails.max() - MDetails.min())
    MDetails = vmin + MDetails * (vmax - vmin)

    velocity = Trends + Details + MDetails
    
    fig, ax = plt.subplots(2,2, figsize=(20,5))
    for i in range(2):
        im = ax[0, i ].imshow(velocity[i,...].T, cmap ='jet', aspect='auto')
        fig.colorbar(im, ax=ax[0,i])
        ax[0,i].set_title(f'Velocity Inline {i}')
    for i in range(2):
        im = ax[1, i ].imshow(velocity[:,i,:].T, cmap ='jet', aspect='auto')
        fig.colorbar(im, ax=ax[1,i])
        ax[1,i].set_title(f'Velocity Xline {i}')
    plt.show()
    