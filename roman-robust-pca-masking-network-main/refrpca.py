import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

class Conv3dC(nn.Module):
    '''
    input: [B, T, H, W] -> output: [B, T, H, W]
    '''

    def __init__(self, kernel, bias=True, gaussian_init=False, sigma=.5):
        super(Conv3dC, self).__init__()
        pad0 = int((kernel[0] - 1) / 2)
        pad1 = int((kernel[1] - 1) / 2)
        self.conv = nn.Conv3d(1, 1, (kernel[0], kernel[1], kernel[1]), (1, 1, 1), (pad0, pad1, pad1), bias=bias, padding_mode='replicate')
        if gaussian_init is True:
            g3d = utils.gaussian_3D(kernel, sigma=sigma, normalized=True)
            self.conv.weight.data[0,0,...] = torch.tensor(g3d)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x.squeeze()


class DU_Layer(nn.Module):
    def __init__(self, kernel, exp_L_initializer, exp_S_initializer, exp_S_side_initializer, coef_L_initializer,
                 coef_S_initializer, coef_S_side_initializer,
                 l1_l1, reweightedl1_l1):
        super(DU_Layer, self).__init__()

        self.conv1 = Conv3dC(kernel)
        self.conv2 = Conv3dC(kernel)
        self.conv3 = Conv3dC(kernel)
        self.conv4 = Conv3dC(kernel)
        self.conv5 = Conv3dC(kernel)
        self.conv6 = Conv3dC(kernel)

        self.exp_L = nn.Parameter(torch.tensor(exp_L_initializer))
        self.exp_S = nn.Parameter(torch.tensor(exp_S_initializer))
        self.exp_S_side = nn.Parameter(torch.tensor(exp_S_side_initializer))
        self.G_conv = Conv3dC((1,7,7), bias=False, gaussian_init=True, sigma=.5)
        self.Z_conv = Conv3dC((1,7,7), bias=False, gaussian_init=True, sigma=.5)
        self.coef_L = nn.Parameter(torch.tensor(coef_L_initializer))
        self.coef_S = nn.Parameter(torch.tensor(coef_S_initializer))
        self.coef_S_side = nn.Parameter(torch.tensor(coef_S_side_initializer))
        self.l1_l1 = l1_l1
        self.reweightedl1_l1 = reweightedl1_l1
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x = data[0]
        L = data[1]
        S = data[2]

        B, T, H, W = x.shape

        Ltmp = self.conv1(x) + self.conv2(L) + self.conv3(S)
        Stmp = self.conv4(x) + self.conv5(L) + self.conv6(S)
        if len(Stmp.shape) == 3: Stmp = Stmp.unsqueeze(0)
        if len(Ltmp.shape) == 3: Ltmp = Ltmp.unsqueeze(0)

        mean_S = torch.mean(torch.abs(S), dim=(1,2,3))
        if self.l1_l1:

            S_zero = torch.zeros((S.shape[0], 1, S.shape[2], S.shape[3])).to(S.device)
            S_side = torch.cat((S_zero, S[:, :-1, ...]), dim=1)
            S_side_motion = self.G_conv(S_side)
            S = self.soft_l1_l1(Stmp, self.coef_S * mean_S, self.coef_S_side * mean_S, S_side_motion)

        elif self.reweightedl1_l1:
            S_zero = torch.zeros((S.shape[0], 1, S.shape[2], S.shape[3])).to(S.device)
            S_side = torch.cat((S_zero, S[:,:-1,...]), dim=1)
            S_side_motion = self.G_conv(S_side)
            if len(S_side_motion.shape) == 3: S_side_motion = S_side_motion.unsqueeze(0)
            S = self.soft_l1_l1_reweighted(Stmp, self.coef_S * mean_S,
                                           self.coef_S_side * mean_S, S_side_motion, self.Z_conv, None)
        else:

            S = self.soft_l1(Stmp, self.coef_S * mean_S)

        L = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)

        L = L.view(B, T, H, W)
        S = S.view(B, T, H, W)

        return (x, L, S)

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

    def soft_l1_l1_reweighted(self, x, w0, w1, alpha1, Z, g):

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



class UnfoldedNet3dC(nn.Module):
    def __init__(self, params=None):
        super(UnfoldedNet3dC, self).__init__()

        self.params = params
        self.filter = self.make_layers()
        self.foreground_classifier = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(1, 3, 3), padding=(0,1,1))
        self.foreground_threshold = nn.Parameter(torch.tensor(0.05))
        self.bn1 = nn.BatchNorm3d(num_features=1)
        self.bn2 = nn.BatchNorm3d(num_features=2)

    def make_layers(self):
        params = self.params
        filt = []
        for i in range(params['layers']):
            filt.append(DU_Layer(kernel=params['kernel'][i],
                                 exp_L_initializer=0.0, exp_S_initializer=0.0, exp_S_side_initializer=0.0,
                                 coef_L_initializer=params['coef_L'],
                                 coef_S_initializer=params['coef_S'],
                                 coef_S_side_initializer=params['coef_S_side'],
                                 l1_l1=params['l1_l1'],
                                 reweightedl1_l1=params['reweightedl1_l1'],))

        return nn.Sequential(*filt)

    def forward(self, x):
        D = x
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1,D.shape[1],1,1).cuda()
        S = x.cuda() - L

        D, L, S = self.filter((D, L, S))

        # Foreground label
        foreground = S.unsqueeze(1)
        y = self.bn1(torch.abs(foreground))
        y = self.foreground_classifier(y)
        y = nn.functional.softmax(y)[:,1,...].squeeze(1)

        return L, S, y

    def get_threshold_params(self):
        params = []
        for layer in self.filter:
            params.append(layer.coef_S)
            params.append(layer.coef_S_side)
            params.append(layer.coef_L)

        return params
