import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='refRPCA_mmnist')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--initial-lr', type=float, default=0.003)
parser.add_argument(
    "--lr_decay_rate", type=float, default=0.3
)  # learning rate decay rate
parser.add_argument(
    "--lr_decay_intv", type=int, default=30
)  # learning rate decay interval
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seed", default=123)
parser.add_argument("--time-steps", default=50)
parser.add_argument("--patch-size", default=None)
parser.add_argument("--max-size", default=128)
parser.add_argument("--crop-size", default=None)
parser.add_argument('--model', default='roman_r')
parser.add_argument("--coeff-L", type=float, default=.1)
parser.add_argument("--coeff-S", type=float, default=0.05)
parser.add_argument("--coeff-Sside", type=float, default=.001)
parser.add_argument("--reweighted", default=False, action='store_true')
parser.add_argument("--l1-l2", default=False, action='store_true')
parser.add_argument('--l1l1', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=75)
parser.add_argument('--checkpoint-dir', default='./ckpt/mnist/mrpcav4')
parser.add_argument('--log-dir', default='./logs/mnist/mrpcav4')
parser.add_argument('--load', default=None, type=str, help='Load and test given checkpoint')
parser.add_argument('--loss_type', default='LM')
parser.add_argument('--clamp-coeff-L', default=.01)
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('-hf', '--hidden-filters', type=int, default=8)

cfgs = parser.parse_args()

cfgs.data_path = '../moving_mnist_fb/seq_32.npy'
cfgs.background_path = '../moving_mnist_fb/background_32.npy'
cfgs.foreground_path = '../moving_mnist_fb/foreground_32.npy'
