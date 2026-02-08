import time
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def gra_forward(density):
    pass


def gra_forward_old(density, size=(40, 40)):
    h, w, d = density.shape[-3:]
    gravity = torch.zeros(density.shape[:-3]+size).cuda()
    t0 = time.time()
    for m in tqdm(range(size[0])):
        for n in range(size[1]):
            gra = 0
            for i in range(h):
                for j in range(w):
                    for k in range(d):
                        gra += density[..., i, j, k] * (50*k+25) * 0.125 / pow(
                            (50*i-100*m-50)**2 + (50*j-100*n-50)**2 + (50*k+25)**2, 1.5)
            gravity[..., m, n] = gra
        # print(m, time.time()-t0)
    # print(gravity[0, 0], gravity.dtype)
    return gravity


class GraForward(nn.Module):

    def __init__(self, in_size=(66, 41, 20), out_size=(66, 41, 5), heights=[0, 200, 400, 600, 800],
                 sample=[100, 100, 100],
                 load_from='work_dirs/gra_forward/accurate.pth'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.sample = sample
        self.param = sample[0] * sample[1] * sample[2] / 1000
        print(in_size, out_size)
        h, w, d = self.in_size[-3:]
        ho, wo, do = self.out_size[-3:]
        in_features = h * w * d
        out_features = ho * wo * do
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features, bias=False)
        self.layer.requires_grad = False
        self.heights = heights
        if os.path.exists(load_from):
            self.load_state_dict(torch.load(
                load_from, map_location=lambda storage, loc: storage))
            print('load forward model')
        elif load_from == '':
            self.set_params()

    def set_params(self):
        h, w, d = self.in_size[-3:]
        ho, wo, do = self.out_size[-3:]
        sh, sw, sd = self.sample
        print(f'h={h}, w={w}, d={d}, ho={ho}, wo={wo}, do={do}, sh={sh}, sw={sw}, sd={sd}')
        sh2, sw2, sd2 = sh / 2, sw / 2, sd / 2

        # Vectorized implementation
        # Create tensors for indices
        i = torch.arange(h, dtype=torch.float32)
        j = torch.arange(w, dtype=torch.float32)
        k = torch.arange(d, dtype=torch.float32)
        m = torch.arange(ho, dtype=torch.float32)
        n = torch.arange(wo, dtype=torch.float32)
        heights = torch.tensor(self.heights, dtype=torch.float32)

        # Calculate distance terms
        # d2g shape: (do, ho, wo, d, h, w)
        # Term related to x: (sh*i - sh*m - sh2)^2
        # i is index 4, m is index 1
        term_x = (sh * i.view(1, 1, 1, 1, h, 1) - sh * m.view(1, ho, 1, 1, 1, 1) - sh2).pow(2)

        # Term related to y: (sw*j - sw*n - sw2)^2
        # j is index 5, n is index 2
        term_y = (sw * j.view(1, 1, 1, 1, 1, w) - sw * n.view(1, 1, wo, 1, 1, 1) - sw2).pow(2)

        # Term related to z: (sd*k + height + sd2)
        # k is index 3, o (heights) is index 0
        dist_z = (sd * k.view(1, 1, 1, d, 1, 1) + heights.view(do, 1, 1, 1, 1, 1) + sd2)
        term_z = dist_z.pow(2)

        # Combine
        denom = (term_x + term_y + term_z).pow(1.5)
        d2g = (dist_z * self.param) / denom

        self.layer.weight.data = d2g.reshape(
            self.out_features, self.in_features)
        self.layer.requires_grad = False

    def forward(self, x):
        # print(x.shape)
        b, c, d, h, w = x.shape
        # print(x.shape)
        x = x.contiguous().view(b, c*d*h*w)
        # print(x.shape, self.layer.weight.shape)
        y = self.layer(x)
        y = y.view(
            b, c, self.out_size[-1], self.out_size[-3], self.out_size[-2])
        return y


class GraForwardConv(nn.Conv2d):

    def __init__(self, in_channels, kernel_size,
                 in_sr=(50, 50, 50), out_sr=(100, 100),
                 set_params=True):
        # sr: sampling rate
        hi_sr, wi_sr, di_sr = in_sr
        ho_sr, wo_sr = out_sr
        stride = (ho_sr//hi_sr, wo_sr//wi_sr)
        assert stride == (2, 2)
        kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        padding = [k//2 for k in kernel_size]
        super().__init__(
            in_channels, 1, kernel_size, stride, padding, bias=False)

        if set_params:
            weight = self.weight.data
            print(weight.shape, self.stride, self.padding, self.kernel_size)
            for i in range(kernel_size[0]):
                for j in range(kernel_size[1]):
                    for k in range(in_channels):
                        weight[0, k, i, j] = (di_sr*k+di_sr/2) * 0.125 / pow(
                            (hi_sr*i-ho_sr*padding[0]+0)**2 +
                            (wi_sr*j-wo_sr*padding[1]+0)**2 +
                            (di_sr*k+di_sr/2)**2,
                            1.5)
            self.weight.data = weight


def train_gra_forward(epoch, batch_size=32, load_from=None):
    accurate = 'work_dirs/gra_forward/accurate.pth'
    in_c, in_h, in_w = 20, 80, 80
    out_h, out_w = 40, 40
    # in_c, in_h, in_w = 2, 8, 8
    # out_h, out_w = 4, 4

    f_model = GraForward((in_h, in_w, in_c), (out_h, out_w)).cuda()
    f_model.eval()
    c_model = GraForwardConv(in_c, 31).cuda()
    if os.path.exists(accurate):
        f_model.load_state_dict(torch.load(accurate))
        f_model = f_model.cuda()
        f_model.eval()
        x = np.array(f_model.layer.weight.data.cpu())
        import matplotlib.pyplot as plt
        print(x.shape, type(x), x.dtype)
        plt.imsave('tools/weight.png', x)
        print('load accurate model')
    else:
        f_model.set_params()
        torch.save(f_model.state_dict(), accurate)
    if load_from and os.path.exists(load_from):
        c_model.load_state_dict(torch.load(load_from))
        print('load convolution model')
        c_model = c_model.cuda()
        c_model.train()

    loss_function = nn.L1Loss()
    optimizer = optim.Adam(c_model.parameters(), lr=1e-4)
    scheduler = MultiStepLR(
        optimizer,
        [int(epoch//5),
         int(epoch//5*2),
         int(epoch//5*3),
         int(epoch//5*4)],
        gamma=0.5)

    for e in range(epoch):
        loss_sum = 0
        loss_comp = 0
        time1 = time2 = 0
        for i in tqdm(range(1000)):
            density = torch.rand(batch_size, in_c, in_h, in_w).cuda()
            t1 = time.time()
            target = f_model(density)
            t2 = time.time()
            pred = c_model(density)
            t3 = time.time()
            time1 += t2 - t1
            time2 += t3 - t2
        #     optimizer.zero_grad()
        #     loss = loss_function(pred*1000, target*1000)
        #     loss.backward()
        #     optimizer.step()
        #     loss_sum += loss.cpu().detach().numpy()
        #     loss_comp += target.abs().mean().cpu().detach().numpy()*1000
        print(time1 / 1000, time2 / 1000)
        # scheduler.step()
        # learn_rate = optimizer.state_dict()['param_groups'][0]['lr']
        # print(f'epoch: {e}\tlr: {learn_rate}\tloss: {loss_sum/1000}'
        #       f'\tcontrast: {loss_comp/1000}\t: {loss}')
        # torch.save(c_model.state_dict(),
        #            f'./work_dirs/gra_forward/c_model_{e}.pth')


if __name__ == '__main__':
    train_gra_forward(1000)
    # , load_from='work_dirs/gra_forward/c_model_1000.pth')
    # import scipy.io as scio
    # import numpy as np

    # c_model = GraForwardConv(5, 33, set_params=True).cuda()

    # # f_model = GraForward((80, 80, 10), (40, 40)).cuda()
    # f_model = GraForward((20, 20, 5), (10, 10)).cuda()
    # f_model.set_params()
    # f_model = f_model.cuda()
    # # # density = scio.loadmat('data/gravity_data/density_1.mat')['density']
    # # # density = density.astype(np.float32)
    # # # density = torch.from_numpy(density).cuda()
    # # # density_ = density.unsqueeze(0).permute(0, 2, 3, 1)
    # torch.manual_seed(327)
    # density_ = torch.rand(1, 5, 20, 20).cuda()
    # # density = density_[0].permute(1, 2, 0)
    # # t0 = time.time()
    # # gra = gra_forward_old(density, size=(10, 10)).unsqueeze(0)
    # # t1 = time.time()
    # gravity = f_model(density_)
    # # # t2 = time.time()
    # gra_conv = c_model(density_)
    # print('basic, max is ', gravity.sum())
    # # print((gra-gravity).abs().sum())
    # print((gravity-gra_conv).abs().sum())
    # # print(t1-t0)
    # # scio.savemat('tools/forward_for.mat', dict(pred=gra))
    # # print(gravity.shape, gra.shape)
    # # print((gravity[0, 0] - gra).abs().sum())
    # # input()

    # # f_model = GraForward((80, 80, 20), (40, 40)).cuda()
    # # density = scio.loadmat('data/gravity_data/density_1.mat')['density']
    # # density = density.astype(np.float32)[np.newaxis, np.newaxis, ...]
    # # density = torch.from_numpy(density).cuda()
    # # print(density.shape, density.dtype)
    # # t0 = time.time()
    # # gravity = f_model(density)
    # # print(time.time()-t0)
    # # print(gravity.shape, gravity.dtype)
    # # gravity = gravity[0, 0].cpu().detach().numpy()
    # # scio.savemat('tools/forward_result.mat', dict(pred=gravity))
    # # g1 = scio.loadmat('data/gravity_data/gra_1.mat')['gra'].T
    # # g2 = scio.loadmat('tools/forward_result.mat')['pred']
    # # g3 = scio.loadmat('tools/forward_for.mat')['pred']
    # # print(g1.max(), g2.max(), g3.max())
    # # print(np.abs(g1-g2).mean())
    # # print(np.abs(g1-g3).max())
    # # print(np.abs(g2-g3).max())
