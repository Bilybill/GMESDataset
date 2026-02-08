from tqdm import tqdm

import torch
import torch.nn as nn


class ForwardGra2D(nn.Module):
    """[n, 1, d, w1] -> [n, 1, w2] 地下二维到地表一维重力反演

    Args:
        nn (_type_): _description_
    """

    def __init__(
            self,
            size,
            sample=(50, 50),
            set_params: str = None,
            load_params: str = None):  # 40000/10
        """_summary_

        Args:
            size (_type_): _description_
            sample (tuple, optional): sample rate in depth and w. 
                Defaults to (50, 50).
            set_params (str, optional): _description_. Defaults to None.
            load_params (str, optional): _description_. Defaults to None.
        """
        super().__init__()

        h, w = size
        w1 = w
        s_i, s_j = sample
        self.in_features = h * w
        self.out_features = w1
        self.out_size = w1
        self.layer = nn.Linear(self.in_features, self.out_features, bias=False)
        self.layer.requires_grad = False

        assert load_params is None or not set_params, \
            'cannot set params and load params simultaneously'
        if set_params:
            weight = self.layer.weight.data.reshape(w1, h, w)
            for m in tqdm(range(w1)):
                for i in range(h):
                    for j in range(w):
                        weight[m, i, j] = (s_i * i + s_i / 2) * 0.125 / pow(
                            (s_j * j - s_j * m) ** 2 +
                            (s_i * i + s_i / 2) ** 2, 1.5)
            self.layer.weight.data = weight.reshape(
                self.out_features, self.in_features)
            torch.save(self.state_dict(), set_params)
        elif load_params:
            self.load_state_dict(torch.load(load_params))
        else:
            Warning('The parameters should be set or loaded')

        # 模型参数不更新
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        b = x.shape[0]
        # print(x.shape)
        x = x.contiguous().view(b, -1)
        # print(x.shape, self.layer.weight.shape)
        y = self.layer(x)
        y = y.view(b, self.out_size)
        return y


class ForwardGra2D2D(nn.Module):
    """[n, 1, d, w1] -> [n, 1, h, w2] 地下二维到地上二维重力反演

    Args:
        nn (_type_): _description_
    """

    def __init__(
            self,
            in_size,
            out_size,
            sample=(50, 50, 500),
            set_params: str = None,
            load_params: str = None):  # 40000/10
        """_summary_

        Args:
            size (_type_): _description_
            sample (tuple, optional): sample rate in depth and w. 
                Defaults to (50, 50).
            set_params (str, optional): _description_. Defaults to None.
            load_params (str, optional): _description_. Defaults to None.
        """
        super().__init__()

        h, w = in_size
        ho, wo = out_size
        assert w == wo, '输入输出w方向采样点数需相同'
        s_i, s_j, s_k = sample
        self.in_features = h * w
        self.out_features = ho * wo
        self.ho = ho
        self.wo = wo
        self.layer = nn.Linear(self.in_features, self.out_features, bias=False)
        self.layer.requires_grad = False

        assert load_params is None or not set_params, \
            'cannot set params and load params simultaneously'
        if set_params:
            weight = self.layer.weight.data.reshape(ho, wo, h, w)
            for n in tqdm(range(wo)):
                for m in range(ho):
                    for i in range(h):
                        for j in range(w):
                            weight[m, n, i, j] = (s_i * i + s_i / 2 + s_k * m) * 0.125 / pow(
                                (s_j * j - s_j * n) ** 2 +
                                (s_i * i + s_i / 2 + s_k * m) ** 2, 1.5)
            self.layer.weight.data = weight.reshape(
                self.out_features, self.in_features)
            torch.save(self.state_dict(), set_params)
        elif load_params:
            self.load_state_dict(torch.load(load_params))
        else:
            Warning('The parameters should be set or loaded')

        # 模型参数不更新
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        b = x.shape[0]
        # print(x.shape)
        x = x.contiguous().view(b, -1)
        # print(x.shape, self.layer.weight.shape)
        y = self.layer(x)
        y = y.view(b, self.ho, self.wo)
        return y


if __name__ == '__main__':

    # gra_forward = ForwardGra2D(
    #     size=[116, 227], load_params='ckpts/forward_models/gra_116_227__50_50.pth')
    # gra = torch.rand((1, 116, 227))
    # den = gra_forward(gra)
    # print(den.shape)

    gra_forward = ForwardGra2D(
        size=[445, 301], sample=(10, 10),
        set_params='ckpts/forward_models/gra_445_301__10_10.pth')
    gra = torch.rand((1, 445, 301))
    den = gra_forward(gra)
    print(den.shape)

    # gra_forward = ForwardGra2D2D(
    #     in_size=[445, 301], out_size=[5, 301],
    #     set_params='ckpts/forward_models/gra_445_301_5__50_50_500.pth')
    # gra = torch.rand((1, 445, 301))
    # den = gra_forward(gra)
    # print(den.shape)
