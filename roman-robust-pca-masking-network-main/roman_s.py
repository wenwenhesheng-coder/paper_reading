import torch
import torch.nn as nn
from torch.autograd import Variable

import utils


class Conv3dC(nn.Module):
    '''
    input: [B, T, H, W] -> output: [B, T, H, W]
    '''

    def __init__(self, kernel, bias=True, gaussian_init=False, sigma=.5, in_channels=1, out_channels=1, return_squeezed=False):
        super(Conv3dC, self).__init__()
        pad0 = int((kernel[0] - 1) / 2)
        pad1 = int((kernel[1] - 1) / 2)
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel[0], kernel[1], kernel[1]), (1, 1, 1), (pad0, pad1, pad1), bias=bias, padding_mode='replicate')
        self.return_squeezed = return_squeezed
        if gaussian_init is True:
            g3d = utils.gaussian_3D(kernel, sigma=sigma, normalized=True)
            self.conv.weight.data[0,0,...] = torch.tensor(g3d)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.conv(x)
        if self.return_squeezed:
            x = x.squeeze(1)
        return x


class ROMAN_S_Layer(nn.Module):
    def __init__(self, kernel, coef_L_initializer,
                 coef_S_initializer, coef_S_side_initializer,
                 l1_l1, reweightedl1_l1, hidden_dim=1, l1_l2=False):
        super(ROMAN_S_Layer, self).__init__()

        self.conv0 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)
        self.conv1 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)
        self.conv2 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)
        self.conv3 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)
        self.conv4 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)
        self.conv5 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)
        self.conv6 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)
        self.conv7 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)
        self.conv8 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)

        self.conv9 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)
        self.conv10 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel)

        self.tau_l = nn.Parameter(torch.tensor(1.0))
        self.tau_s = nn.Parameter(torch.tensor(1.0))

        self.rho = nn.Parameter(torch.tensor(1.0))
        self.mask_scaling = nn.Parameter(torch.tensor(20.0))

        self.G_conv = Conv3dC((1,7,7), bias=False, gaussian_init=True, sigma=.5, return_squeezed=True)

        self.Z_conv = Conv3dC((1,7,7), bias=False, gaussian_init=True, sigma=.5, return_squeezed=True)

        self.coef_L = nn.Parameter(torch.tensor(coef_L_initializer))
        self.coef_S = nn.Parameter(torch.tensor(coef_S_initializer))
        self.coef_S_side = nn.Parameter(torch.tensor(coef_S_side_initializer))
        self.l1_l1 = l1_l1
        self.reweightedl1_l1 = reweightedl1_l1
        self.l1_l2 = l1_l2
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, data):
        # data = (input, low-rank, mask, multiplier)
        X = data[0]
        L = data[1]
        M = data[2]
        U = data[3]

        B, T, H, W = X.shape

        # L branch
        background_mask = 1 - self.conv0(M)
        background_mask_squared = torch.square(background_mask)
        _L = L - self.tau_l*self.conv2(background_mask_squared * self.conv1(L))
        _X = self.tau_l*self.conv3(background_mask_squared * X.unsqueeze(1))
        _U = -self.tau_l/(self.rho+1e-4) * self.conv4(background_mask * U)
        Ltmp = _L + _X + _U
        Ltmp = Ltmp.squeeze(1)

        Ltmp = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)
        L = Ltmp.view(B, T, H, W)

        # S branch

        background_mask = 1 - self.conv6(M)
        foreground = X.unsqueeze(1) - self.conv5(L)
        foreground_squared = torch.square(foreground)
        _M = M + self.tau_s*self.conv8(foreground_squared * background_mask)
        _U = - self.tau_s/(self.rho + 1e-4) * self.conv7(foreground * U)
        Mtmp = _M + _U
        Mtmp = Mtmp.squeeze(1)

        mean_M = torch.mean(torch.abs(Mtmp), dim=(1, 2, 3))
        if self.l1_l1:
            M_zero = torch.zeros((M.shape[0], 1, M.shape[2], M.shape[3])).to(M.device)
            M_side = torch.cat((M_zero, M[:, :-1, ...]), dim=1)
            M_side_motion = self.G_conv(M_side)
            Mtmp = self.soft_l1_l1(Mtmp, self.coef_S * mean_M, self.coef_S_side * mean_M, M_side_motion)

        elif self.reweightedl1_l1:
            M_zero = M[:,0,...].unsqueeze(1)
            M_side = torch.cat((M_zero, Mtmp[:, :-1, ...]), dim=1)
            M_side_motion = self.G_conv(M_side)
            if len(M_side_motion.shape) == 3: M_side_motion = M_side_motion.unsqueeze(0)
            Mtmp = self.soft_l1_l1_reweighted(Mtmp, self.coef_S * mean_M,
                                           self.coef_S_side * mean_M, M_side_motion, self.Z_conv, None)
        elif self.l1_l2:
            M_zero = M[:, 0, ...].unsqueeze(1)
            M_side = torch.cat((M_zero, Mtmp[:, :-1, ...]), dim=1)
            M_side_motion = self.G_conv(M_side)
            if len(M_side_motion.shape) == 3: M_side_motion = M_side_motion.unsqueeze(0)

            Mtmp = self.soft_l1(Mtmp + self.coef_S_side * (M_side_motion - M), self.coef_S * mean_M)
        else:

            Mtmp = self.soft_l1(Mtmp, self.coef_S * mean_M)

        Mtmp = nn.functional.sigmoid((Mtmp-.5)*self.mask_scaling)

        M = Mtmp.view(B, T, H, W)

        # U branch
        U = U + self.rho*(1 - self.conv9(M)) * (self.conv10(L) - X.unsqueeze(1))

        return (X, L, M, U)

    def svtC(self, x, th):
        U, S, V = torch.svd(x)
        S = torch.sign(S) * self.relu(torch.abs(S) - nn.functional.relu(th) * torch.abs(S[:, 0]).unsqueeze(1))
        S_diag = torch.diag_embed(S, dim1=-2, dim2=-1)
        return torch.matmul(torch.matmul(U, S_diag), V.transpose(2, 1))

    def mixthre(self, x, th):
        x_norm = torch.norm(x, p=2, dim=1)
        return self.relu(1 - th / x_norm)[:, None] * x

    def soft_l1(self, z, th):
        B, T, H, W = z.shape
        z = z.view(B, T, H * W)
        th = th.unsqueeze(-1).unsqueeze(-1).repeat(1, z.shape[1], z.shape[2])
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th)
        return out

    def soft_l1_l1(self, z, w0, w1, alpha1):

        B, T, H, W = z.shape
        z = z.view(B, T, H * W)
        alpha1 = alpha1.view(B, T, H * W)

        alpha0 = torch.zeros(alpha1.size(), device=z.device, dtype=z.dtype)
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)

        w0 = w0.unsqueeze(-1).unsqueeze(-1).repeat(1, z.shape[1], z.shape[2])
        w1 = w1.unsqueeze(-1).unsqueeze(-1).repeat(1, z.shape[1], z.shape[2])

        w0_sorted = torch.where(condition, w0, w1)
        w1_sorted = torch.where(condition, w1, w0)

        cond1 = z >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = z >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = z >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = z >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = z - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = z - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = z + w0_sorted + w1_sorted
        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5))))

    def soft_l1_l1_reweighted(self, x, w0, w1, alpha1, Z, g=None):

        B, T, H, W = x.shape
        x = Z(x).view(B, T, H * W)
        alpha1 = alpha1.view(B, T, H * W)

        alpha0 = torch.zeros(alpha1.size(), device=x.device, dtype=x.dtype)
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)

        w0 = w0.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], x.shape[2])
        w1 = w1.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], x.shape[2])

        if g is not None:
            w0_sorted = torch.where(condition, w0, w1) * g
            w1_sorted = torch.where(condition, w1, w0) * g
        else:
            w0_sorted = torch.where(condition, w0, w1)
            w1_sorted = torch.where(condition, w1, w0)

        cond1 = x >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = x >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = x >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = x >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = x - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = x - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = x + w0_sorted + w1_sorted

        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5))))


class ROMAN_S(nn.Module):
    def __init__(self, params=None):
        super(ROMAN_S, self).__init__()

        self._earlystop=None
        self.hd = params['hidden_filters']
        self.params = params
        self.filter = self.make_layers()

    def make_layers(self):
        params = self.params
        filt = []
        for i in range(params['layers']):
            filt.append(ROMAN_S_Layer(kernel=params['kernel'][i],
                                      coef_L_initializer=params['coef_L'],
                                      coef_S_initializer=params['coef_S'],
                                      coef_S_side_initializer=params['coef_S_side'],
                                      l1_l1=params['l1_l1'],
                                      reweightedl1_l1=params['reweightedl1_l1'],
                                      hidden_dim=self.hd,
                                      l1_l2=params['l1_l2']))

        return nn.Sequential(*filt)

    def forward(self, x):
        D = x
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()

        M = torch.where(torch.abs(x.cuda() - L) > .05, torch.ones_like(L) * 0.5, torch.zeros_like(L))
        # M = torch.abs(x.cuda() - L)

        U = torch.zeros((L.shape[0], self.hd, L.shape[1], L.shape[2], L.shape[3])).cuda()

        if self._earlystop is None:
            D, L, M, U = self.filter((D, L, M, U))
        else:
            for i in range(self._earlystop):
                D, L, M, U = self.filter[i].forward((D, L, M, U))

        return L, None, M



    def get_threshold_params(self):
        params = []
        for layer in self.filter:
            params.append(layer.coef_S)
            params.append(layer.coef_S_side)
            params.append(layer.coef_L)

        return params

    def set_early(self, es):
        self._earlystop = es