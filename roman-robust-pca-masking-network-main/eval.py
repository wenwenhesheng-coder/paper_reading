import os
import numpy as np
import torch
import yaml

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from main_roman import eval, run_one_batch
from refrpca import UnfoldedNet3dC
from roman_s import ROMAN_S
from roman_r import ROMAN_R
from unet3d.unet3DC import UNet


from dataloader.video_loader import Loader as LoaderTransform
from utils import stats_per_frame, compute_F1

import argparse

parser = argparse.ArgumentParser(description='Model evaluator')
parser.add_argument('-p', '--path', type=str, default=None)
parser.add_argument('-c', '--category', type=str, default=None)
parser.add_argument('-f', '--full', default=False, action='store_true', help='Evaluate on whole clip (all in val)')

args = parser.parse_args()

def eval_mnist(net, data_loader, cfgs):
    batch_size = cfgs.batch_size

    net.eval()
    with torch.no_grad():
        eval_losses = []
        mse_L = []
        F1 = []

        '''F1thresholds = list(np.arange(.1, 1.0, step=.1))
        trainstats = []

        # for batch_idx in range(data_loader.train_samples // batch_size):
        for batch_idx in range(10):
            batch = data_loader.load_batch_train(batch_size)

            losses, outputs = run_one_batch(net, batch, cfgs)
            loss_tot, lD, lL, lS, stats = losses
            for i in range(batch_size):
                stats = [F1_scores((outputs[2][i].detach().cpu().numpy() > tr).astype(np.int),
                                  (np.swapaxes(batch[1], 0, 1)[i] > 0.2).astype(np.int)) for tr in F1thresholds]
                trainstats.append(stats)

        F1train = np.asarray(F1train)'''

        while True:

            batch = data_loader.load_batch_test(batch_size)
            if len(batch[0]) == 0:
                break

            losses, outputs = run_one_batch(net, batch, cfgs)

            loss_tot, lD, lL, lS, stats = losses

            eval_losses.append(loss_tot.item())
            for i in range(batch_size):
                stats = stats_per_frame((outputs[2][i].detach().cpu().numpy() > 0.5).astype(np.int), (np.swapaxes(batch[1], 0, 1)[i] > 0.2).astype(np.int))
                F1.append(compute_F1([stats]))

                o = outputs[0][i].detach().cpu().numpy()
                o[o<0] = 0.0
                o[o>1] = 1.0
                mse_L.append(np.square(o - np.swapaxes(batch[2], 0, 1)[i]).mean())

        toprint = 'loss: {:.6f}, L: {:.6f}, F1: {:.6f}'.format(
            np.mean(eval_losses),
            np.mean(mse_L),
            np.mean(F1))
        print(toprint)

        return toprint




subdir = args.path
assert os.path.exists(subdir), 'Specified directory does not exist'

results = {}
results_text = ''
raw_res = []
pre = []
rec = []

data_loader = None

cfg_file = os.path.join(subdir, 'config.yaml')
with open(cfg_file) as myfile:
    cfgs = yaml.load(myfile, Loader=yaml.Loader)

if args.category is not None:
    cfgs.categories = [args.category]


if data_loader is None:
    data_loader = LoaderTransform(cfgs.data_path, cfgs.categories, time_steps=cfgs.time_steps,
                                  flatten=False, scale=False, maxsize=cfgs.max_size,
                                  patch_size=cfgs.patch_size, seed=cfgs.seed, crop_size=cfgs.crop_size,
                                  compute_weights=False, include_in_train=cfgs.include_in_train,
                                  split=cfgs.split, all_in_val=args.full)
    '''data_loader = Moving_MNIST_RPCA_Loader(cfgs.data_path, path_fg=cfgs.foreground_path,
                                           path_bg=cfgs.background_path,
                                           time_steps=20, flatten=False, scale=True)'''
else:
    data_loader.update_split(cfgs.split) # Disable this for MMNIST
    data_loader.current_idx_val = 0
    data_loader.current_idx_test = 0

dataset = 'cdnet2014'
# dataset = 'mmnist'

params_net = {'layers': cfgs.layers,
              'kernel': [(1, 5)] * 10,
              'hidden_filters': cfgs.hidden_filters,
              'coef_L': cfgs.coeff_L,
              'coef_S': cfgs.coeff_S,
              'coef_S_side': cfgs.coeff_Sside,
              'l1_l1': cfgs.l1l1,
              'reweightedl1_l1': cfgs.reweighted,
              'img_size': cfgs.crop_size,
              'l1_l2': cfgs.l1_l2,
              }
params_net['kernel'] = params_net['kernel'][0:params_net['layers']]


if cfgs.model == 'roman_s':
    net = ROMAN_S(params_net)
elif cfgs.model == 'roman_r':
    net = ROMAN_R(params_net)
elif cfgs.model == 'unet3d':
    net = UNet(1, 2, False)
else:
    net = UnfoldedNet3dC(params_net)


if os.path.exists(os.path.join(subdir, 'checkpoint.pth')) is False:
    pass
net.load_state_dict(torch.load(os.path.join(subdir, 'checkpoint.pth')))

#### Counting parameters ####
pytorch_total_params = sum(p.numel() for p in net.parameters())
print('Model has {} trainable parameters'.format(pytorch_total_params))

if torch.cuda.is_available():
    net = net.cuda()


# Enable this for MMNIST
# out = eval_mnist(net, data_loader, cfgs)

# Enable this for CDNet
f1t, res, probs, gtprobs, pr_rec, res_train = eval(net, data_loader, cfgs, subdir, None, None, verbose=False) #----------
# f1t, res, probs, gtprobs, pr_rec, res_train = eval(net, data_loader, cfgs, None, None, None, verbose=False) #---------- Without visual outputs
raw_res.append(res[0][f1t]) #---------


results_text += subdir + ': Evaluating on ' + args.category + '\n'
results_text += 'Best F1 on training set: {}\n'.format(res_train[0][f1t])
results_text += 'layers: {}, lr:{:.4f}, L:{:.4f} S:{:.4f}, Ss:{:.4f}\n'.format(cfgs.layers, cfgs.initial_lr, cfgs.coeff_L,
                                                                               cfgs.coeff_S, cfgs.coeff_Sside)

# Enable this for MMNIST
# results_text += out+'\n'

pre.append(pr_rec[0][f1t][0])
rec.append(pr_rec[0][f1t][1])
results_text += 'pr: {} rec: {}'.format(pr_rec[0][f1t][0], pr_rec[0][f1t][1])

'''auc = roc_auc_score(np.concatenate(gtprobs), np.concatenate(probs))
fpr, tpr, _ = roc_curve(np.concatenate(gtprobs), np.concatenate(probs))
np.savez_compressed(os.path.join(subdir, 'fpr'), fpr)
np.savez_compressed(os.path.join(subdir, 'tpr'), tpr)
with open(os.path.join(subdir, 'auc.txt'), 'w') as myfile:
    myfile.write(str(auc))'''

print(results_text)
print(','.join(['{:.6f}'.format(s) for s in raw_res]))
print('F1: {:.6f}'.format(np.mean(raw_res)))



