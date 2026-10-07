import sys
import torch

print("Python:", sys.executable, flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")
    print("当前使用 CPU，仅测试代码是否能运行", flush=True)

    # 兼容项目中写死的 .cuda()
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

from roman_s import ROMAN_S
from roman_r import ROMAN_R

params = {
    "layers": 2,
    "kernel": [(1, 5), (1, 5)],
    "hidden_filters": 4,
    "coef_L": 0.1,
    "coef_S": 0.05,
    "coef_S_side": 0.001,
    "l1_l1": False,
    "reweightedl1_l1": True,
    "l1_l2": False,
    "img_size": None,
}

D = torch.rand(1, 5, 16, 16, device=device)

for name, model_class in [
    ("ROMAN-S", ROMAN_S),
    ("ROMAN-R", ROMAN_R),
]:
    print(f"\nTesting {name}...", flush=True)

    model = model_class(params).to(device)
    L, S, M = model(D)

    print("Input :", tuple(D.shape), flush=True)
    print("L     :", tuple(L.shape), flush=True)
    print("Mask  :", tuple(M.shape), flush=True)
    print("Mask range:", M.min().item(), M.max().item(), flush=True)
    print("Finite:", torch.isfinite(M).all().item(), flush=True)

    loss = L.mean() + M.mean()
    loss.backward()

    print(f"{name} forward/backward: OK", flush=True)

print("\n全部测试完成", flush=True)