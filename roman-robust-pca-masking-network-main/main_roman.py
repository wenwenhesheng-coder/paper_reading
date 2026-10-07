import cv2
from cv2 import VideoWriter_fourcc

from dataloader.video_loader import Loader as LoaderTransform
from roman_s import ROMAN_S
from roman_r import ROMAN_R
from refrpca import UnfoldedNet3dC as RefRPCA
from unet3d.unet3DC import UNet as Unet3D

import numpy as np
from dataloader.moving_mnist_data_loader import Moving_MNIST_RPCA_Loader
import time
import datetime


import torch
import torch.nn as nn
import os
import yaml
import tqdm

from utils import stats_per_frame, compute_F1, compute_pre_rec, TverskyLoss

mse_loss = nn.MSELoss()
cce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
mae_loss = nn.L1Loss()
tversky = TverskyLoss(alpha_t=0.5, beta_t=0.5)
mtloss = None
mtloss2 = None


def run_one_batch(net, batch, cfgs, nof1=False, profile=False, F1thresholds=None, class_weights=None):

    if cfgs.dataset == "mmnist":
        D, S, L = batch
        roi = np.ones((cfgs.batch_size, cfgs.time_steps, D.shape[-2], D.shape[-1]))
        D = torch.Tensor(np.swapaxes(D, 0, 1)).cuda()
        L = torch.Tensor(np.swapaxes(L, 0, 1)).cuda()
        S = torch.Tensor(np.swapaxes(S, 0, 1)).cuda()
        # Creating binary mask based on non-zero values of digits
        M = (S > 0.2).long().cuda()

    else:
        D, M, roi_batch = batch
        D = D.float().cuda()
        D.requires_grad = False

        if np.random.randint(2)==1:
            M_motion = (M==255)
        else:
            M_motion = torch.logical_or(M==255, M==170)

        # M_motion = torch.logical_or(M==255, M==170)
        M_motion = M_motion.long().cuda()
        M_motion.requires_grad = False

        L = None
        S = None

    # if profile is False:
    if profile is False:
        outputs_L, outputs_S, outputs_M = net(D)
    else:
        with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
            outputs_L, outputs_S, outputs_M = net(D)
        print(prof.key_averages().table(sort_by='cuda_time_total'))

    if cfgs.loss_type == 'L_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        if cfgs.dataset == 'mmnist':
            loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float()) + mse_loss(mask * outputs_L, mask * D)
        else:
            if class_weights is not None:
                _w = torch.where(M_motion==0, class_weights[0], class_weights[1])
                _w.requires_grad = False
                loss_tot = cfgs.alpha * _w * bce_loss(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)
            else:
                loss_tot = cfgs.alpha * bce_loss(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)

    elif cfgs.loss_type == 'L_tversky':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        if cfgs.dataset == 'mmnist':
            loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float()) + mse_loss(mask * outputs_L, mask * D)
        else:
            loss_tot = cfgs.alpha * tversky(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)

    elif cfgs.loss_type == 'L_tversky_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        if cfgs.dataset == 'mmnist':
            loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float()) + mse_loss(mask * outputs_L, mask * D)
        else:
            loss_tot = 0.5 * tversky(outputs_M, M_motion.float()) + 0.5* bce_loss(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)

    elif cfgs.loss_type == 'tversky_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        if cfgs.dataset == 'mmnist':
            loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float())
        else:
            loss_tot = 0.5 * tversky(outputs_M, M_motion.float()) + 0.5* bce_loss(outputs_M, M_motion.float())

    #### The following losses are useful for UNet: ####
    elif cfgs.loss_type == 'LM':
        loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float()) + mse_loss(outputs_L, L)

        if cfgs.loss_type == 'tversky_bce':
            mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
            if cfgs.dataset == 'mmnist':
                loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float())
            else:
                loss_tot = 0.5 * tversky(outputs_M, M_motion.float()) + 0.5 * bce_loss(outputs_M, M_motion.float())


    elif cfgs.loss_type == 'M_cce':
            mask = 1 - torch.logical_or(M == 0, M == 85).long().cuda().detach()
            loss_tot = cce_loss(outputs_M, mask)
            outputs_M = torch.nn.functional.softmax(outputs_M, dim=1)[:, 1]

    elif cfgs.loss_type == 'tversky_cce':
            mask = 1 - torch.logical_or(M == 0, M == 85).long().cuda().detach()
            loss_tot = 0.5 * cce_loss(outputs_M, mask)
            # loss_tot = 0.5 * cce_loss(outputs_M, M_motion)
            outputs_M = torch.nn.functional.softmax(outputs_M, dim=1)[:, 1]
            loss_tot = loss_tot + 0.5 * tversky(outputs_M, mask)
            # loss_tot = loss_tot + 0.5 * tversky(outputs_M, M_motion.float())


    if nof1 is False:

        if F1thresholds is None:
            stats = stats_per_frame((outputs_M.detach().cpu().numpy() > 0.5).astype(np.int),
                                    M.detach().cpu().numpy())
        else:
            stats = [stats_per_frame((outputs_M.detach().cpu().numpy() > tr).astype(np.int),
                                     M.detach().cpu().numpy()) for tr in F1thresholds]

    else:
        stats = (1, 1, 1, 1)

    return (loss_tot, stats), (outputs_L, outputs_S, outputs_M)


def train(net, data_loader, cfgs, log_dir, log_file, ckpt):

    global cce_loss
    epochs = cfgs.epochs
    batch_size = cfgs.batch_size

    if cfgs.dataset != 'mmnist':
        cce_loss = nn.CrossEntropyLoss(weight=torch.tensor(data_loader.class_weights).float().cuda())
    else:
        cce_loss = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(list(net.parameters()), lr=cfgs.initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfgs.lr_decay_intv,
                                                gamma=cfgs.lr_decay_rate)


    for epoch in range(epochs):

        print('Trainig epoch {}'.format(epoch))

        data_loader.current_idx_train = 0
        data_loader.current_idx_val = 0

        ##### TRAIN #####

        net.train()
        train_losses = []

        iterator = tqdm.tqdm(range(data_loader.train_samples//batch_size))
        for batch_idx in iterator:

            optimizer.zero_grad()

            batch = data_loader.load_batch_train(batch_size)

            losses, outputs = run_one_batch(net, batch, cfgs, nof1=True)

            loss_tot, f1 = losses
            loss_tot.backward()

            train_losses.append(loss_tot.item())

            if 'unet' not in cfgs.model:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)

            optimizer.step()

            # Clamp coefficients
            if 'unet' not in cfgs.model:
                for i, filter in enumerate(net.filter):
                    filter.coef_L.data = filter.coef_L.clamp(min=cfgs.clamp_coeff_L)
                    filter.coef_S.data = filter.coef_S.clamp(min=1e-4)
                    filter.coef_S_side.data = filter.coef_S_side.clamp(min=1e-4)

            toprint = 'loss: {:.6f}'.format(
                np.mean(train_losses))
            iterator.set_description(toprint)

        data_loader.shuffle()

        scheduler.step()


        ##### TEMPORARY VALIDATION #####


        if cfgs.dataset == 'mmnist':
            toprint = eval_mnist(net, data_loader)
            with open(log_file, 'a') as myfile:
                myfile.write(toprint + '\n')

        elif (epoch+1)%30==0 and cfgs.split != -1.0:
            net.eval()
            with torch.no_grad():

                eval_losses = []
                F1 = []

                batch_idx = 0
                while True:

                    batch = data_loader.load_batch_validation(batch_size)
                    if len(batch[0]) == 0:
                        break

                    losses, outputs = run_one_batch(net, batch, cfgs, profile=False, nof1=True)

                    loss_tot, f1 = losses

                    eval_losses.append(loss_tot.item())
                    F1.append(f1)

                    batch_idx += 1

                toprint = 'epoch {} -- eval loss: {:.6f}'.format(
                    epoch,
                    np.mean(eval_losses))
                print(toprint)
                with open(log_file, 'a') as myfile:
                    myfile.write(toprint+'\n')


    torch.save(net.state_dict(), ckpt)

    if cfgs.dataset == 'mmnist':
        f1_test = eval_mnist(net, data_loader)
    else:
        f1_test = eval(net, data_loader, cfgs, log_dir, log_file, ckpt)


    return f1_test


def eval_mnist(net, data_loader):
    batch_size = cfgs.batch_size

    net.eval()
    with torch.no_grad():
        eval_losses = []
        F1 = []

        while True:

            batch = data_loader.load_batch_validation(batch_size)
            if len(batch[0]) == 0:
                break

            losses, outputs = run_one_batch(net, batch, cfgs)

            loss_tot, stats = losses

            eval_losses.append(loss_tot.item())
            for i in range(batch_size):
                stats = stats_per_frame((outputs[2][i].detach().cpu().numpy() > 0.5).astype(np.int), (np.swapaxes(batch[1], 0, 1)[i] > 0.2).astype(np.int))
                F1.append(compute_F1([stats]))

        toprint = 'loss: {:.6f}, F1: {:.6f}'.format(
            np.mean(eval_losses),
            np.mean(F1))
        print(toprint)

        return toprint

def eval(net, data_loader, cfgs, log_dir, log_file, ckpt, verbose=True, threshVid=None):

    # Evaluating on real videos, with visual output and statistics computed on a video-basis (if the selected dataset has multiple videos)

    if verbose: print('Testing on videos separately:')
    net.eval()

    all_losses = []
    all_F1_train = []
    all_F1_eval = []
    all_pr_eval = []

    gt_probs = []
    probs = []

    cat_idx = 0

    with torch.no_grad():

        counter = 0
        counter2 = 0
        # F1thresholds = list(np.arange(.1, 1.1, step=.1,))
        F1thresholds = [0.5]

        # Looping over categories
        while True:

            clip_idx = 0
            eval_losses = []
            train_stats = []
            eval_stats = []

            # Looping over clips
            while True:

                batch = data_loader.load_clip_from_category_eval(cat_idx, clip_idx)
                if batch is None:
                    # Meaning we are out of categories
                    break
                if len(batch[0]) == 0:
                    # Meaning we are out of clips in category
                    break

                if (cat_idx, clip_idx) not in data_loader.eval:
                    clip_idx += 1
                    continue

                if counter2==20:
                    break
                else:
                    counter2 += 1



                D, M, roi = batch

                # Forward pass

                losses, outputs = run_one_batch(net, batch, cfgs, nof1=True)

                L_assembled = outputs[0]
                M_assembled = outputs[2]

                # Visual output
                if cfgs.dataset == 'mmnist':
                    inputs = batch[0][:,0, ...].astype(np.float32)
                else:
                    inputs = batch[0][0, ...].detach().cpu().numpy().astype(np.float32)
                outputs_L = L_assembled[0, ...]
                outputs_M = M_assembled[0, ...]

                '''if 'cce' in cfgs.loss_type:
                    # outputs_M = M_assembled[0, 1, ...]
                    outputs_M = torch.argmax(M_assembled, dim=1)[0,...].float()'''

                if log_dir is not None:# Visual output
                    L_image = outputs_L[10, ...].detach().cpu().numpy()
                    L_image = cv2.cvtColor(L_image*255, cv2.COLOR_GRAY2BGR)
                    D_image = cv2.cvtColor(inputs[10,...]*255, cv2.COLOR_GRAY2BGR)

                    cv2.imwrite(os.path.join(log_dir, '{}_L.png'.format(counter)),
                                L_image)
                    cv2.imwrite(os.path.join(log_dir, '{}_D.png'.format(counter)),
                                D_image)
                    cv2.imwrite(os.path.join(log_dir, '{}_M.png'.format(counter)),
                                    cv2.cvtColor(outputs_M[10, ...].detach().cpu().numpy()*255, cv2.COLOR_GRAY2BGR)
                                    )


                if log_dir is not None and (cat_idx, clip_idx) in data_loader.eval and threshVid is not None:
                    fourcc = VideoWriter_fourcc(*'MP4V')
                    out = cv2.VideoWriter(os.path.join(log_dir, 'output{}.mp4'.format(counter)), fourcc, 15.0, (inputs.shape[2]*5, inputs.shape[1]))
                    for i in range(inputs.shape[0]):
                        frame = np.concatenate([inputs[i]*255,
                                                M[0,i].detach().cpu().numpy(),
                                                outputs_L[i].detach().cpu().numpy()*255,
                                                outputs_M[i].detach().cpu().numpy()*255,
                                                (outputs_M[i].detach().cpu().numpy()>F1thresholds[threshVid])*255.0], axis=1)
                        frame[frame>255.0] = 255.0
                        frame[frame<0.0] = 0.0
                        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        out.write(frame)
                    out.release()

                counter += 1


                # Statistics
                loss_tot, f1 = losses

                stats = [stats_per_frame((outputs_M.unsqueeze(0).detach().cpu().numpy() >= tr).astype(np.int),
                                         M.detach().cpu().numpy()) for tr in F1thresholds]

                if (cat_idx, clip_idx) in data_loader.eval:
                    eval_losses.append(loss_tot.item())
                    eval_stats.append(stats)

                    # FOR ROC
                    masks_target = M[0].detach().cpu().numpy()
                    mask_gt = (masks_target == 255).astype(np.int)  # F1 score computed for WHITE pixels (motion)
                    roi_main = np.logical_and(masks_target != 85, masks_target != 170).astype(
                        np.int)  # ROI does not accounts for ROI and NON-UNKNOWN pixels

                    gt_probs.append(mask_gt[roi_main == True])
                    probs.append(outputs_M.detach().cpu().numpy()[roi_main == True])
                if (cat_idx, clip_idx) in data_loader.train:
                    train_stats.append(stats)

                clip_idx += 1

            if batch is None:
                # Meaning we are out of categories
                break

            catF1_train = [compute_F1([batch_stats[f1_index] for batch_stats in train_stats]) for f1_index in range(len(F1thresholds))]
            catF1_eval = [compute_F1([batch_stats[f1_index] for batch_stats in eval_stats]) for f1_index in range(len(F1thresholds))]
            all_F1_train.append(catF1_train)
            all_F1_eval.append(catF1_eval)
            all_pr_eval.append([
                compute_pre_rec([batch_stats[f1_index] for batch_stats in eval_stats]) for f1_index in range(len(F1thresholds))
            ])

            catLoss = np.mean(eval_losses)
            all_losses.append(catLoss)

            toprint = 'Video {} -- loss: {:.6f} '.format(cat_idx, catLoss) + ', '.join(['F1 ({}): {:.6f}'.format(F1thresholds[i], catF1_eval[i]) for i in range(len(F1thresholds))])
            if verbose:
                print(toprint)
                with open(log_file, 'a') as myfile:
                    myfile.write(toprint + '\n')

            best_trainF1_idx = np.argmax(catF1_train)
            if verbose: print('Best F1 on training set: {}'.format(catF1_train[best_trainF1_idx]))
            if verbose: print('Best F1 obtained with threshold {}: F1= {}'.format(F1thresholds[best_trainF1_idx], catF1_eval[best_trainF1_idx]))

            cat_idx += 1

        totLoss = np.mean(all_losses)
        totF1 = np.mean(np.asarray(all_F1_eval), axis=0)
        toprint = 'Overall -- loss: {:.6f} '.format(totLoss) + ', '.join(['F1 ({}): {:.6f}'.format(F1thresholds[i], totF1[i]) for i in range(len(F1thresholds))])
        if verbose:
            print(toprint)
            with open(log_file, 'a') as myfile:
                myfile.write(toprint + '\n')

        return best_trainF1_idx, all_F1_eval, probs, gt_probs, all_pr_eval, all_F1_train



if __name__ == '__main__':

    ##### Main configuration #####
    # dataset = 'mmnist'
    dataset = 'cdnet2014'

    if dataset == "mmnist":
        from configs.mmnist import cfgs
    elif dataset == "cdnet2014":
        from configs.cdnet import cfgs
    else:
        assert False, 'Invalid dataset'
    cfgs.dataset = dataset

    if 'model' not in cfgs:
        cfgs.model = 'roman_s'



    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)

    params_net = {'layers': cfgs.layers,
                  'kernel': [(1, 5)] * 10,
                  'hidden_filters': cfgs.hidden_filters,
                  'coef_L': cfgs.coeff_L,
                  'coef_S': cfgs.coeff_S,
                  'coef_S_side': cfgs.coeff_Sside,
                  'l1_l1': False,
                  'reweightedl1_l1': cfgs.reweighted,
                  'l1_l2': cfgs.l1_l2,
                  'img_size': cfgs.crop_size,
                  }
    params_net['kernel'] = params_net['kernel'][0:params_net['layers']]

    # Log files
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M')
    if cfgs.load is None:
        log_dir = os.path.join(cfgs.log_dir, st)
        duplicate_counter = 1

        log_dir_2 = log_dir
        while os.path.exists(log_dir_2) is True:
            duplicate_counter += 1
            log_dir_2 = log_dir + '_' + str(duplicate_counter)
        log_dir = log_dir_2
        os.makedirs(log_dir)

        log_file = os.path.join(log_dir, 'log_{}'.format(st))
        with open(log_file, 'a') as myfile:
            myfile.write(str(cfgs))

        checkpoint_file = os.path.join(log_dir, 'checkpoint.pth')
        cfgs_file = os.path.join(log_dir, 'config.yaml')
        with open(cfgs_file, 'w') as myfile:
            yaml.dump(cfgs, myfile, default_flow_style=False)


    ##### Model #####
    if cfgs.model == 'roman_s':
        net = ROMAN_S(params_net)
    elif cfgs.model == 'roman_r':
        net = ROMAN_R(params_net)
    elif cfgs.model == 'refrpca':
        net = RefRPCA(params_net)
    elif cfgs.model == 'unet3d':
        # cfgs.loss_type = 'tversky'
        cfgs.initial_lr = 0.001

        net = Unet3D(1, 2, bilinear=False)

        state_dict = torch.load('./unet3d/unet_carvana_scale1.0_epoch2.pth')
        for name, W in state_dict.items():
            if name == 'inc.double_conv.0.weight':
                state_dict[name] = W[:, 0].unsqueeze(1).unsqueeze(2).repeat_interleave(3, dim=2)
            elif len(W.shape) > 1:
                if '.up.' not in name and name != 'outc.conv.weight':
                    # print(name)
                    state_dict[name] = W.unsqueeze(2).repeat_interleave(3, dim=2)
                else:
                    state_dict[name] = W.unsqueeze(2)
        net.load_state_dict(state_dict)

    if torch.cuda.is_available():
        net = net.cuda()

    #### Counting parameters ####
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('Model has {} trainable parameters'.format(pytorch_total_params))

    ##### Data #####
    data_loader = None
    if dataset == 'mmnist':
        data_loader = Moving_MNIST_RPCA_Loader(cfgs.data_path, path_fg=cfgs.foreground_path, path_bg=cfgs.background_path,
                                               time_steps=20, flatten=False, scale=True)
    elif dataset == 'cdnet2014':
        data_loader = LoaderTransform(cfgs.data_path, cfgs.categories, time_steps=cfgs.time_steps,
                             flatten=False, scale=False, maxsize=cfgs.max_size,
                             patch_size=cfgs.patch_size, seed=cfgs.seed, crop_size=cfgs.crop_size,
                             compute_weights=True, include_in_train=cfgs.include_in_train, split=cfgs.split)
    else:
        assert False, 'Invalid dataset for data_loader'


    ##### Train #####
    f1 = train(net, data_loader, cfgs, log_dir, log_file, checkpoint_file)


