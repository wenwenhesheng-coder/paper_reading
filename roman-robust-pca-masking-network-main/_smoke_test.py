# CPU smoke test for ROMAN-S / ROMAN-R / RefRPCA (unfolded net)
# The models hardcode .cuda(); on a CPU-only torch build we monkeypatch it away.
import torch

torch.Tensor.cuda = lambda self: self  # no-op cuda for CPU testing

from roman_s import ROMAN_S
from roman_r import ROMAN_R
from refrpca import UnfoldedNet3dC


def check(name, net, D, target=None):
    net.train()
    L, S, M = net(D)
    assert L.shape == D.shape, f'{name}: L shape {L.shape} != {D.shape}'
    assert M.shape == D.shape, f'{name}: M shape {M.shape} != {D.shape}'
    ok_range = bool((M >= 0).all() and (M <= 1).all())
    print(f'{name}: L {tuple(L.shape)}, M {tuple(M.shape)}, M range OK = {ok_range}, '
          f'finite = {torch.isfinite(M).all().item()}')
    loss = ((M - target).pow(2)).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    print(f'{name}: loss {loss.item():.4f}, grads OK = {grads_ok}')
    return M


torch.manual_seed(0)
B, T, H, W = 1, 5, 16, 16
D = torch.rand(B, T, H, W)
target = (torch.rand(B, T, H, W) > 0.7).float()

base = {'layers': 2, 'kernel': [(1, 5)] * 2, 'hidden_filters': 4,
        'coef_L': 0.1, 'coef_S': 0.05, 'coef_S_side': 0.001,
        'l1_l1': False, 'reweightedl1_l1': True, 'l1_l2': False,
        'img_size': None}

for name, cls in [('ROMAN-S (reweighted l1l1)', ROMAN_S),
                  ('ROMAN-R (reweighted l1l1)', ROMAN_R),
                  ('RefRPCA (unfolded)', UnfoldedNet3dC)]:
    try:
        net = cls(base)
        check(name, net, D, target)
    except Exception as e:
        import traceback
        print(f'{name}: FAILED -> {type(e).__name__}: {e}')
        traceback.print_exc()

# also check other prox branches
for name, kw in [('ROMAN-S plain l1', {'l1_l1': False, 'reweightedl1_l1': False}),
                 ('ROMAN-S l1_l2', {'l1_l1': False, 'reweightedl1_l1': False, 'l1_l2': True}),
                 ('ROMAN-R plain l1', {'l1_l1': False, 'reweightedl1_l1': False}),
                 ('ROMAN-R l1_l2', {'l1_l1': False, 'reweightedl1_l1': False, 'l1_l2': True})]:
    p = dict(base)
    p.update(kw)
    cls = ROMAN_S if name.startswith('ROMAN-S') else ROMAN_R
    try:
        check(name, cls(p), D, target)
    except Exception as e:
        import traceback
        print(f'{name}: FAILED -> {type(e).__name__}: {e}')
        traceback.print_exc()

print('smoke test done')
