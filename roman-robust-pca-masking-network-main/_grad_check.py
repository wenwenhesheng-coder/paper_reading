import torch
torch.Tensor.cuda = lambda self: self

from roman_s import ROMAN_S
from roman_r import ROMAN_R
from refrpca import UnfoldedNet3dC

torch.manual_seed(0)
B, T, H, W = 1, 5, 16, 16
D = torch.rand(B, T, H, W)
target = (torch.rand(B, T, H, W) > 0.7).float()

base = {'layers': 2, 'kernel': [(1, 5)] * 2, 'hidden_filters': 4,
        'coef_L': 0.1, 'coef_S': 0.05, 'coef_S_side': 0.001,
        'l1_l1': False, 'reweightedl1_l1': True, 'l1_l2': False,
        'img_size': None}


def report(name, net):
    net.train()
    L, S, M = net(D)
    ((M - target).pow(2)).mean().backward()
    print(f'--- {name} ---')
    for n, p in net.named_parameters():
        if p.grad is None:
            print(f'  [no grad] {n}')
        elif not torch.isfinite(p.grad).all():
            print(f'  [NON-FINITE grad] {n}  {p.grad.abs().max().item():.3e}')
        elif p.grad.abs().max() == 0:
            print(f'  [zero grad] {n}')
    print('  (all other params have finite, nonzero grads)')


report('ROMAN-S reweighted', ROMAN_S(dict(base)))
report('ROMAN-R reweighted', ROMAN_R(dict(base)))
report('ROMAN-R plain l1', ROMAN_R(dict(base, reweightedl1_l1=False)))
report('RefRPCA', UnfoldedNet3dC(dict(base)))
